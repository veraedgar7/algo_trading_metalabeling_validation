//+------------------------------------------------------------------+
//|                                                    Peregrine.mq5 |
//|                                      Copyright 2026, Edgar Vera.  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Edgar Vera."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict

#resource "\\Files\\Raven_for_Peregrine.onnx" as uchar onnx_data[]
#include <Trade/Trade.mqh>
CTrade trade;

// STRATEGY INPUTS
input int    Magic_Number       = 240326;
input double Risk_Percent       = 1;
input int    BB_Period          = 20;
input double BB_Deviation       = 1.5;
input int    ATR_Period         = 14;
input double ATR_Mult_SL        = 1.5;
input double ATR_Mult_TP        = 5;
input double ATR_Trailing_Mult  = 5.0;
input double BE_Plus_Mult       = 0.3;
input int    Max_Daily_Trades   = 4;
input double Max_Daily_Loss_Pct = 4.0; // maximum allowed daily loss
input int    Max_Spread_Pips    = 200;

input int    Start_Hour         = 9;    // Server time (London open)
input int    Exit_Hour_Server   = 22;   // Server time (New York close)
input bool   Trade_On_Friday    = false; // Avoid weekend gaps
input int    leverage           = 1;
// MACRO FILTER
input int    MA_H4_Period       = 20;
input int    ADX_Period         = 14;
input int    ADX_Threshold      = 25;
input double Vol_Mult_Factor    = 1.2;   // Relative volume filter, 20% above average

int bbHandle, atrHandle, maH4Handle, adxHandle, maHandle, rsiHandle;
// Variables to keep track of the last entry

datetime last_trade_time = 0; // Initialized at 0
ulong    last_opened_ticket = 0;
int stype = 1;
long onnx_handle = INVALID_HANDLE;
#define FEATURES_COUNT 9

int OnInit() {
   trade.SetExpertMagicNumber(Magic_Number);
   bbHandle = iBands(_Symbol, _Period, BB_Period, 0, BB_Deviation, PRICE_CLOSE);
   atrHandle = iATR(_Symbol, _Period, ATR_Period);
   rsiHandle   = iRSI(_Symbol, _Period, 14, PRICE_CLOSE);
   maHandle = iMA(_Symbol, _Period, 200, 0, MODE_SMA, PRICE_CLOSE);
   maH4Handle = iMA(_Symbol, PERIOD_H4, MA_H4_Period, 0, MODE_SMA, PRICE_CLOSE);
   adxHandle = iADXWilder(_Symbol, PERIOD_H4, ADX_Period);
   InitONNX();
   return(INIT_SUCCEEDED);
}

// AI MODEL INITIALIZATION
void InitONNX() {
   onnx_handle = OnnxCreateFromBuffer(onnx_data, ONNX_DEFAULT);

   if(onnx_handle != INVALID_HANDLE) {
      const long input_shape[] = {1, 8};
      if(!OnnxSetInputShape(onnx_handle, 0, input_shape)) Print("Error setting input shape");

      const long output_shape[] = {1};
      if(!OnnxSetOutputShape(onnx_handle, 0, output_shape)) Print("Error setting output shape");

      Print("AI MODEL: SYNCHRONIZED.");
   }
}


// QUERY THE MODEL
int PredictAI(double bb_norm, double adx, double rsi, double vol_rel, double h_sin, double h_cos, double dist_ma, double stretch) {
   if(onnx_handle == INVALID_HANDLE) return 1;

   float input_data[8]; // Input array
   input_data[0] = (float)bb_norm;
   input_data[1] = (float)adx;
   input_data[2] = (float)rsi;
   input_data[3] = (float)vol_rel;
   input_data[4] = (float)h_sin;
   input_data[5] = (float)h_cos;
   input_data[6] = (float)dist_ma;
   input_data[7] = (float)stretch;

   // Output: predicted label, no probability
   long output_data[1];
   output_data[0] = -1;

   // Run inference
   if(!OnnxRun(onnx_handle, ONNX_DEFAULT, input_data, output_data)) {
      Print("ONNX EXECUTION ERROR: Code ", GetLastError());
      return 1; // Safety fallback if the model fails
   }

   return (int)output_data[0];
}

// TRADING WINDOW FUNCTION
bool IsInTradingWindow() {
   MqlDateTime dt;
   TimeCurrent(dt);

   // Do not trade on weekends
   if(dt.day_of_week == 0 || dt.day_of_week == 6) return false;

   // Friday filter (optional, to avoid swap/gap exposure)
   if(!Trade_On_Friday && dt.day_of_week == 5 && dt.hour > 15) return false;

   // Trading hours filter
   if(dt.hour < Start_Hour || dt.hour >= Exit_Hour_Server) return false;

   return true;
}

// H4 ADX TREND STRENGTH FUNCTION
bool TrendStrengthADX() {
   double adxBuffer[];
   ArraySetAsSeries(adxBuffer, true);
   if(CopyBuffer(adxHandle, 0, 1, 1, adxBuffer) < 1) return false;
   return (adxBuffer[0] >= ADX_Threshold);
}

// BREAKOUT VOLUME VALIDATION FUNCTION
bool IsVolumeValid() {
   // Volume of the 30M candle that just closed
   long currentVol = iTickVolume(_Symbol, _Period, 1);
   // Average of the last 20 candles
   long candl = 20;
   long sumVol = 0;
   for(int i=1; i<=candl; i++) {
      sumVol += iTickVolume(_Symbol, _Period, i);
   }
   double avgVol = (double)sumVol / candl;
   return (currentVol > (avgVol * Vol_Mult_Factor));
}

// DAILY PROTECTION FUNCTION
bool CheckDailyProtections() {
   MqlDateTime t; TimeToStruct(TimeCurrent(), t);
   static int lastDay = -1;
   static int tradesToday = 0;
   static double initialDailyBalance = 0;

   // Reset counters at the start of a new day
   if(t.day != lastDay) {
      tradesToday = 0;
      initialDailyBalance = AccountInfoDouble(ACCOUNT_BALANCE);
      lastDay = t.day;
   }

   // 1. Daily loss control
   double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   double lossThreshold = initialDailyBalance * (1.0 - (Max_Daily_Loss_Pct / 100.0));

   if(currentEquity <= lossThreshold) {
      Comment("HARD STOP: Daily loss limit reached. Trading suspended.");
      return false;
   }

   // 2. Trade count control
   datetime dayStart = StringToTime(IntegerToString(t.year)+"."+IntegerToString(t.mon)+"."+IntegerToString(t.day)+" 00:00");
   HistorySelect(dayStart, TimeCurrent());
   int totalHistory = HistoryDealsTotal();
   tradesToday = 0;

   for(int i=0; i<totalHistory; i++) {
      ulong ticket = HistoryDealGetTicket(i);
      if(HistoryDealGetInteger(ticket, DEAL_MAGIC) == Magic_Number && HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT)
         if(HistoryDealGetDouble(ticket, DEAL_PROFIT) < 0) tradesToday++;
   }

   if(tradesToday >= Max_Daily_Trades) {
      Comment("LIMIT REACHED: Already executed " + (string)Max_Daily_Trades + " trades today.");
      return false;
   }

   return true;
}

// NEWS SHIELD FUNCTION
bool IsHighImpactNewsSoon() {
   MqlCalendarValue values[];
   // Search range: 5 minutes before and 5 minutes after
   datetime from = TimeCurrent() - 300;
   datetime to   = TimeCurrent() + 300;

   // Filter USD news only
   if(CalendarValueHistory(values, from, to, "US")) {
      for(int i=0; i<ArraySize(values); i++) {
         MqlCalendarEvent event;
         // Get the event detail by its ID
         if(CalendarEventById(values[i].event_id, event)) {
            // CALENDAR_IMPORTANCE_HIGH = red folder (high impact news)
            if(event.importance == CALENDAR_IMPORTANCE_HIGH) {
               Print("BLOCK ACTIVE: ", event.name, " detected in calendar.");
               return true;
            }
         }
      }
   }
   return false;
}

void ManageExits() {

   if(!PositionSelect(_Symbol) || PositionGetInteger(POSITION_MAGIC) != Magic_Number) return;

   double bid    = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask    = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double entry  = PositionGetDouble(POSITION_PRICE_OPEN);
   double sl     = PositionGetDouble(POSITION_SL);
   double tp     = PositionGetDouble(POSITION_TP);
   double profit = PositionGetDouble(POSITION_PROFIT);
   double swap   = PositionGetDouble(POSITION_SWAP);
   int    type   = (int)PositionGetInteger(POSITION_TYPE);
   datetime openTime = (datetime)PositionGetInteger(POSITION_TIME);

   MqlDateTime dt;
   TimeCurrent(dt);

   // SESSION CLOSE STRATEGY
   if(dt.hour >= Exit_Hour_Server) {
      if(profit > 0) {
         // If in profit at NY close, secure breakeven + swap
         // Calculate the price equivalent of the accumulated swap to compensate for it
         double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
         double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
         double volume    = PositionGetDouble(POSITION_VOLUME);

         // Price adjustment to cover negative swap (if any)
         double swapOffset = (swap < 0) ? (MathAbs(swap) / (volume * (tickValue / tickSize))) : 0;
         double newSL = (type == POSITION_TYPE_BUY) ? (entry + swapOffset) : (entry - swapOffset);

         // Only move if the new stop loss is better than the current one
         if((type == POSITION_TYPE_BUY && newSL > sl) || (type == POSITION_TYPE_SELL && (newSL < sl || sl == 0))) {
            // intentionally left as a hook for a future breakeven+swap adjustment
         }
      }
   }

   // EXPIRATION STRATEGY
   if(TimeCurrent() - openTime >= 86400) { // 24 hours
      if(profit <= 0) {
         trade.PositionClose(_Symbol);
         Print("EXPIRATION: Closing losing trade after 24h.");
         return;
      }
   }

   // TRAILING MANAGEMENT (ATR)
   double atr[]; ArraySetAsSeries(atr, true);
   if(CopyBuffer(atrHandle, 0, 0, 1, atr) <= 0) return;

   if(type == POSITION_TYPE_BUY) {
      double nSL = NormalizeDouble(bid - (atr[0] * ATR_Trailing_Mult), _Digits);
      if(bid > entry + (atr[0] * 2.5) && nSL > sl) trade.PositionModify(_Symbol, nSL, tp);
   }
   else {
      double nSL = NormalizeDouble(ask + (atr[0] * ATR_Trailing_Mult), _Digits);
      if(ask < entry - (atr[0] * 2.5) && (nSL < sl || sl == 0)) trade.PositionModify(_Symbol, nSL, tp);
   }
}

void OnTick() {
   ManageExits();
   if(GlobalPositionExists(_Symbol)) return;

   // If high impact news is expected in the next 5 minutes, or happened 5 minutes ago, do not enter.
   if(IsHighImpactNewsSoon()) {Comment("STATE: News block active. Standing by...");return;}
   // If the daily trade limit or daily loss limit has been reached
   if(!CheckDailyProtections()) return;

   // SPREAD VALIDATION
   int currentSpread = (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   if(currentSpread > Max_Spread_Pips) {
      Comment("BLOCKED: Spread exceeds the allowed limit ",Max_Spread_Pips," (", currentSpread, ") pts");
      return;
   }

   // Only open a trade if there is trend strength and enough volume
   if(!TrendStrengthADX()) {return;}
   if(!IsVolumeValid()) {return;}
   Comment("STATE: Trading");

   // 1. Get Bollinger Bands and ATR data
   double upper[], lower[], atr[];
   ArraySetAsSeries(upper, true); ArraySetAsSeries(lower, true); ArraySetAsSeries(atr, true);
   if(CopyBuffer(bbHandle, 1, 0, 1, upper) <= 0 || CopyBuffer(bbHandle, 2, 0, 1, lower) <= 0 || CopyBuffer(atrHandle, 0, 0, 1, atr) <= 0) return;

   // 2. Get H4 trend
   double maH4[]; ArraySetAsSeries(maH4, true);
   if(CopyBuffer(maH4Handle, 0, 0, 1, maH4) <= 0) return;
   double closeH4 = iClose(_Symbol, PERIOD_H4, 1);

   // 3. Current prices
   double c_close = iClose(_Symbol, _Period, 1); // Close of the previous candle
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK), bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   // DYNAMIC ENTRY LOGIC
   bool Buy  = (c_close > upper[0] && closeH4 > maH4[0]);
   bool Sell = (c_close < lower[0] && closeH4 < maH4[0]);

   if((Buy || Sell)) {
      // Indicators for the AI model
      double ma_ia[], upper_ia[], lower_ia[], adx_ia[], rsi_ia[];
      ArraySetAsSeries(ma_ia, true); ArraySetAsSeries(upper_ia, true); ArraySetAsSeries(lower_ia, true); ArraySetAsSeries(adx_ia, true); ArraySetAsSeries(rsi_ia, true);

      if(CopyBuffer(maHandle, 0, 1, 1, ma_ia)  < 1 ||
         CopyBuffer(bbHandle, 1, 1, 1, upper_ia)  < 1 ||
         CopyBuffer(bbHandle, 2, 1, 1, lower_ia)  < 1 ||
         CopyBuffer(adxHandle, 0, 1, 1, adx_ia) < 1 ||
         CopyBuffer(rsiHandle, 0, 1, 1, rsi_ia)  < 1 ) return;

      // RELATIVE VOLUME FOR THE AI MODEL
      long cur_vol = iTickVolume(_Symbol, _Period, 1);
      long sum_vol = 0;
      for(int i=2; i<=21; i++) sum_vol += iTickVolume(_Symbol, _Period, i);

      // HOUR ENCODED AS SINE AND COSINE
      datetime candleTime = iTime(_Symbol, _Period, 1);
      MqlDateTime dt;
      TimeToStruct(candleTime, dt);

      // EXHAUSTION CANDLE
      double current_atr = atr[0];
      double candle_range = MathAbs(iHigh(_Symbol, _Period, 1) - iLow(_Symbol, _Period, 1));

      // AI model inputs
      double bbn = (c_close - lower_ia[0]) / (upper_ia[0] - lower_ia[0]);
      double current_adx = adx_ia[0];
      double current_rsi = rsi_ia[0];
      double vol_rel = (sum_vol > 0) ? (double)cur_vol / ((double)sum_vol / 20.0) : 1;
      double h_sin = sin(2.0 * M_PI * dt.hour / 24.0);
      double h_cos = cos(2.0 * M_PI * dt.hour / 24.0);
      double dist = (ma_ia[0] != 0) ? (c_close - ma_ia[0]) / ma_ia[0] : 0;
      double candle_stretch = (current_atr > 0) ? (candle_range / current_atr) : 1.0; // A value of 3.0 indicates a giant candle (possible news/panic event)

      MqlDateTime time_now;
      TimeCurrent(time_now);

      int decision = PredictAI(bbn, current_adx, current_rsi, vol_rel, h_sin, h_cos, dist, candle_stretch);

      if(decision == 1){
         // BUY: close above the upper band + bullish H4 trend
         if(Buy) {
            double sl = NormalizeDouble(ask - (atr[0] * ATR_Mult_SL), _Digits);
            double tp = NormalizeDouble(ask + (atr[0] * ATR_Mult_TP), _Digits);
            double lot = CalculateLot((ask-sl)/_Point);
            SendNotification("SHIELD BUY: Lot " + DoubleToString(lot, 2) + " | TP: " + DoubleToString(tp, 2));
            trade.Buy(lot, _Symbol, ask, sl, tp, "Shield_Alpha");
         }
         // SELL: close below the lower band + bearish H4 trend
         else if(Sell) {
            double sl = NormalizeDouble(bid + (atr[0] * ATR_Mult_SL), _Digits);
            double tp = NormalizeDouble(bid - (atr[0] * ATR_Mult_TP), _Digits);
            double lot = CalculateLot((sl-bid)/_Point);
            SendNotification("SHIELD SELL: Lot " + DoubleToString(lot, 2) + " | TP: " + DoubleToString(tp, 2));
            trade.Sell(lot, _Symbol, bid, sl, tp, "Shield_Alpha");
         }
      }
   }
}

double CalculateLot(double stopPoints) {
   if(stopPoints <= 0) return 0.01;
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskMoney = balance * Risk_Percent / 100.0;

   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double pointValuePerLot = (tickValue / tickSize) * _Point;
   double lot = riskMoney / (stopPoints * pointValuePerLot);

   double marginRequired;
   if(OrderCalcMargin(ORDER_TYPE_BUY, _Symbol, lot, SymbolInfoDouble(_Symbol, SYMBOL_ASK), marginRequired)) {
      if(marginRequired > (balance * leverage)) {
         lot = (balance * leverage) / (marginRequired / lot);
         Print("NOTICE: Lot adjusted due to MARGIN limit");
      }
   }

   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   lot = MathFloor(lot / step) * step;
   return MathMax(SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN), lot);
}

void OnTradeTransaction(const MqlTradeTransaction& trans,const MqlTradeRequest& request,const MqlTradeResult& result){
   // When a position closes (DEAL)
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD) {
      // Select the last deal from history
      if(HistoryDealSelect(trans.deal)) {
         // Confirm this is our position closing (not opening)
         long entry_type = HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
         if(entry_type == DEAL_ENTRY_OUT) {
            if(HistoryDealGetString(trans.deal, DEAL_SYMBOL) == _Symbol) {
               ulong position_id  = HistoryDealGetInteger(trans.deal, DEAL_POSITION_ID);
               double profit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT);
               double swap   = HistoryDealGetDouble(trans.deal, DEAL_SWAP);
               double comm   = HistoryDealGetDouble(trans.deal, DEAL_COMMISSION);
               string res = ((profit + swap + comm) > 0) ? "PROFIT" : "LOSS";
               SendNotification("SHIELD CLOSED: " + res + " | Profit: $" + DoubleToString(profit, 2)+ " | Swap: $" + DoubleToString(swap, 2)+ " | Commission: $" + DoubleToString(comm, 2));
               last_trade_time = TimeCurrent();
            }
         }
      }
   }
}

bool GlobalPositionExists(string symbol) {
    for(int i = PositionsTotal() - 1; i >= 0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(PositionSelectByTicket(ticket)) {
            if(PositionGetString(POSITION_SYMBOL) == symbol) return true;
        }
    }
    return false;
}
