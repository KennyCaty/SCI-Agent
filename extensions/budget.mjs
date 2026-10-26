// Classify request duration when it settles; transport timeouts are separate.
export class BudgetClock {
  constructor(limitSeconds, now = () => performance.now()) {
    this.limit = limitSeconds; this.now = now; this.started = null;
    this.active = null; this.failedSeconds = 0; this.calls = [];
    this.activeExcludedSeconds=0;this.retryExclusions=[];
  }
  start() { if (this.started !== null) throw new Error('Budget already started'); this.started = this.now(); }
  beginCall() {
    if (this.active !== null) throw new Error('Overlapping model calls in one Pi session');
    this.active = this.now();
    this.activeExcludedSeconds=0;
  }
  excludeRetrySpan(record) {
    if(this.active===null)throw new Error('Retry exclusion outside model call');
    if(!Number.isFinite(record.seconds)||record.seconds<0)throw new Error('Invalid retry exclusion');
    this.failedSeconds+=record.seconds;this.activeExcludedSeconds+=record.seconds;
    this.retryExclusions.push(record);
  }
  finishCall(outcome, cancelledByCaller = false) {
    if (this.active === null) throw new Error('No active model call');
    const seconds = Math.max(0, (this.now()-this.active)/1000);
    const excluded = ['error','aborted'].includes(outcome) && !cancelledByCaller;
    if (excluded) this.failedSeconds += Math.max(0,seconds-this.activeExcludedSeconds);
    this.calls.push({outcome,seconds,excluded_from_normal:excluded,cancelled_by_caller:cancelledByCaller,
      retry_excluded_seconds:this.activeExcludedSeconds,excluded_seconds:excluded?seconds:this.activeExcludedSeconds,
      normal_seconds:Math.max(0,seconds-(excluded?seconds:this.activeExcludedSeconds))});
    this.active = null;
  }
  snapshot() {
    const wall = this.started === null ? 0 : Math.max(0,(this.now()-this.started)/1000);
    return {wall_seconds:wall,failed_api_seconds:this.failedSeconds,
      normal_seconds:Math.max(0,wall-this.failedSeconds),api_call_pending:this.active !== null};
  }
  exhausted() {
    // Count a failed request separately before checking the normal-time budget.
    return this.started !== null && this.active === null && this.snapshot().normal_seconds >= this.limit;
  }
}

export function watchBudget(clock, onExhausted, intervalMs = 100) {
  return setInterval(() => { if (clock.exhausted()) onExhausted(); }, intervalMs);
}
