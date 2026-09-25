"use strict";

const byId = (id) => document.getElementById(id);

const state = {
  socket: null,
  authenticated: false,
  phase: "idle",
  wallets: [],
  walletBalance: null,
  positions: [],
  orders: [],
  blacklist: [],
  summary: null,
  authTimer: null,
};

const phaseLabels = {
  idle: "待机",
  starting: "启动中",
  running: "运行中",
  stopping: "停止中",
  killed: "Kill Switch",
};

const errorLabels = {
  "Invalid license key": "License Key 无效",
  "License expired": "License 已过期",
  "Key already in use": "该 License 已被另一个客户端使用",
  "License server unreachable": "无法连接 License 服务",
  insufficient_balance: "钱包余额不足，或 bankroll 高于钱包余额",
  no_wallet_registered: "没有注册执行钱包",
  farm_already_running: "Farm 已经在运行",
  "Failed to save wallet": "钱包数据保存失败",
  "Failed to save blacklist": "黑名单保存失败",
};

function websocketUrl() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws`;
}

function socketOpen() {
  return state.socket && state.socket.readyState === WebSocket.OPEN;
}

function setConnection(kind, label) {
  const pill = byId("connection-pill");
  pill.className = `status-pill ${kind}`;
  byId("connection-label").textContent = label;
}

function setPhase(phase) {
  state.phase = phase;
  const label = phaseLabels[phase] || phase;
  const badge = byId("farm-state-badge");
  badge.textContent = label;
  badge.className = `badge ${phase === "idle" || phase === "killed" ? "neutral" : phase}`;
  byId("metric-state").textContent = label;

  const locked = ["starting", "running", "stopping"].includes(phase);
  document
    .querySelectorAll("#farm-form input:not([type='checkbox']), #farm-form select")
    .forEach((element) => {
      element.disabled = locked;
    });
  byId("risk-confirm").disabled = locked;
  byId("stop-button").disabled = !["starting", "running"].includes(phase);
  updateStartAvailability();
}

function translateError(reason) {
  if (!reason) return "未知错误";
  if (reason.startsWith("undeployed deposit wallet")) {
    return "Deposit Wallet 尚未部署，请先登录 Polymarket 完成账户钱包创建";
  }
  return errorLabels[reason] || reason;
}

function send(payload) {
  if (!socketOpen()) {
    showToast("WebSocket 未连接", "error");
    return false;
  }
  state.socket.send(JSON.stringify(payload));
  return true;
}

function connect(licenseKey) {
  if (state.socket && state.socket.readyState < WebSocket.CLOSING) return;

  setConnection("connecting", "连接中");
  byId("connect-button").disabled = true;
  appendLog("连接", "正在连接本地后端…");

  const socket = new WebSocket(websocketUrl());
  state.socket = socket;

  socket.addEventListener("open", () => {
    appendLog("连接", "WebSocket 已打开，正在验证 License");
    send({ type: "auth", license_key: licenseKey });
    state.authTimer = window.setTimeout(() => {
      if (!state.authenticated) {
        showToast("License 验证超时", "error");
        socket.close();
      }
    }, 12000);
  });

  socket.addEventListener("message", (event) => {
    let message;
    try {
      message = JSON.parse(event.data);
    } catch (_error) {
      appendLog("协议", "收到无法解析的服务器消息", "error");
      return;
    }
    handleMessage(message);
  });

  socket.addEventListener("error", () => {
    showToast("WebSocket 连接失败，请确认后端正在运行", "error");
  });

  socket.addEventListener("close", () => {
    window.clearTimeout(state.authTimer);
    const wasActive = ["starting", "running", "stopping"].includes(state.phase);
    state.authenticated = false;
    state.socket = null;
    setConnection("offline", "未连接");
    byId("disconnect-button").disabled = true;
    byId("connect-button").disabled = false;
    byId("login-panel").hidden = false;
    byId("dashboard").hidden = true;
    setPhase("idle");
    appendLog("断开", wasActive ? "连接中断；后端将尝试停止 Farm 并清理订单" : "连接已关闭", wasActive ? "warning" : "info");
    if (wasActive) showToast("连接中断，请到 Polymarket 检查是否仍有挂单", "error");
  });
}

function disconnect() {
  if (!state.socket) return;
  if (["starting", "running", "stopping"].includes(state.phase)) {
    const confirmed = window.confirm(
      "断开连接会停止 Farm。后端会尝试撤单，但你仍需要检查 Polymarket。确定断开吗？",
    );
    if (!confirmed) return;
  }
  state.socket.close(1000, "user disconnect");
}

function handleMessage(message) {
  switch (message.type) {
    case "auth_ok":
      window.clearTimeout(state.authTimer);
      state.authenticated = true;
      setConnection("online", "已连接");
      byId("disconnect-button").disabled = false;
      byId("login-panel").hidden = true;
      byId("dashboard").hidden = false;
      appendLog("认证", "License 验证成功", "success");
      showToast("已连接交易后端", "success");
      updateStartAvailability();
      break;

    case "auth_fail":
      window.clearTimeout(state.authTimer);
      appendLog("认证", translateError(message.reason), "error");
      showToast(translateError(message.reason), "error");
      setConnection("offline", "认证失败");
      break;

    case "wallet_list":
      state.wallets = Array.isArray(message.wallets) ? message.wallets : [];
      renderWallet();
      appendLog("钱包", `已载入 ${state.wallets.length} 个钱包`);
      break;

    case "wallet_error":
      appendLog("请求", translateError(message.reason), "error");
      showToast(translateError(message.reason), "error");
      if (state.phase === "starting") setPhase("idle");
      break;

    case "farm_started":
      setPhase("running");
      appendLog("Farm", "策略任务已启动，正在初始化和筛选市场", "success");
      showToast("Farm 已启动", "success");
      break;

    case "farm_cancelled":
      setPhase("idle");
      appendLog("Farm", "策略已停止，撤单清理流程已返回", "success");
      showToast("Farm 已停止，请复核 Polymarket 挂单", "success");
      break;

    case "farm_error":
      setPhase("idle");
      appendLog("Farm", translateError(message.reason), "error");
      showToast(translateError(message.reason), "error");
      break;

    case "farm_killed":
      setPhase("killed");
      appendLog(
        "Kill",
        `达到亏损限制：净亏损 ${formatMoney(message.net_loss)}，策略已终止`,
        "error",
      );
      showToast("Kill Switch 已触发，请立即检查订单和持仓", "error");
      break;

    case "farm_summary":
      state.summary = message;
      renderSummary();
      break;

    case "farm_positions":
      state.positions = Array.isArray(message.positions) ? message.positions : [];
      renderPositions();
      break;

    case "order_placed":
      addOrderEvent(message, "挂单中");
      appendLog(
        "挂单",
        `${message.slug} ${message.outcome} ${message.side} @ ${message.price} × ${message.size}`,
        "success",
      );
      break;

    case "order_cancelled":
      markOrderCancelled(message);
      appendLog("撤单", `${message.slug} ${message.outcome}：${message.reason}`, "warning");
      break;

    case "order_filled":
      addOrderEvent(message, "已成交");
      appendLog(
        "成交",
        `${message.slug} ${message.outcome} ${message.side} @ ${message.price} × ${message.size}`,
        "warning",
      );
      showToast(`${message.slug} ${message.outcome} 已成交`, "warning");
      break;

    case "blacklist_list":
      state.blacklist = Array.isArray(message.markets) ? message.markets : [];
      renderBlacklist();
      break;

    case "blacklist_error":
      appendLog("黑名单", translateError(message.reason), "error");
      showToast(translateError(message.reason), "error");
      break;

    default:
      appendLog("消息", `收到 ${message.type || "unknown"}`);
  }
}

function currentWallet() {
  return state.wallets.length ? state.wallets[0] : null;
}

function parseAmount(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function renderWallet() {
  const wallet = currentWallet();
  byId("wallet-empty").hidden = Boolean(wallet);
  byId("wallet-details").hidden = !wallet;

  if (!wallet) {
    state.walletBalance = null;
    byId("metric-balance").textContent = "—";
    updateStartAvailability();
    return;
  }

  state.walletBalance = parseAmount(wallet.balance);
  byId("wallet-address").textContent = wallet.proxy_address;
  byId("wallet-address").title = wallet.proxy_address;
  byId("wallet-slot").textContent = wallet.wallet_id;

  if (state.walletBalance === null) {
    byId("metric-balance").textContent = "不可用";
    byId("wallet-balance-state").textContent = "RPC 读取失败";
    byId("wallet-balance-state").className = "negative";
  } else {
    byId("metric-balance").textContent = formatMoney(state.walletBalance);
    byId("wallet-balance-state").textContent = formatMoney(state.walletBalance);
    byId("wallet-balance-state").className = state.walletBalance > 0 ? "positive" : "";
  }
  updateStartAvailability();
}

function renderSummary() {
  const summary = state.summary;
  if (!summary) return;
  const reportedBalance = parseAmount(summary.wallet_balance);
  if (reportedBalance !== null) {
    state.walletBalance = reportedBalance;
    byId("metric-balance").textContent = formatMoney(reportedBalance);
  }
  byId("metric-markets").textContent = String(summary.active_markets ?? 0);
  byId("metric-volume").textContent = `成交量 ${formatMoney(summary.total_volume)}`;
  byId("metric-rewards-day").textContent = formatMoney(summary.rewards_per_day);
  byId("metric-rewards-total").textContent = `累计 ${formatMoney(summary.total_rewards)}`;
  byId("metric-loss").textContent = formatMoney(summary.session_loss);
  byId("metric-loss-limit").textContent = `上限 ${formatMoney(summary.max_session_loss)}`;
  byId("metric-elapsed").textContent = formatDuration(summary.elapsed_seconds);
  updateStartAvailability();
}

function renderPositions() {
  const body = byId("positions-body");
  body.replaceChildren();
  byId("position-count").textContent = `${state.positions.length} 个市场`;

  if (!state.positions.length) {
    body.append(emptyRow(5, "暂无持仓"));
    return;
  }

  for (const position of state.positions) {
    const row = document.createElement("tr");
    const market = document.createElement("td");
    market.className = "market-cell";
    const title = document.createElement("strong");
    title.textContent = position.question || position.slug;
    const slug = document.createElement("small");
    slug.textContent = position.slug || position.market_id;
    market.append(title, slug);
    row.append(
      market,
      textCell(`${formatNumber(position.yes_price, 3)} / ${formatNumber(position.no_price, 3)}`, "mono"),
      textCell(`${formatNumber(position.yes_shares, 2)} / ${formatNumber(position.no_shares, 2)}`, "mono"),
      textCell(formatMoney(position.capital_deployed), "mono"),
      pnlCell(position.unrealized_pnl),
    );
    body.append(row);
  }
}

function addOrderEvent(message, status) {
  state.orders.unshift({
    ...message,
    status,
    timestamp: new Date(),
    display_id: message.order_id || `${Date.now()}-${Math.random()}`,
  });
  state.orders = state.orders.slice(0, 100);
  renderOrders();
}

function markOrderCancelled(message) {
  const existing = state.orders.find((order) => order.order_id === message.order_id);
  if (existing) {
    existing.status = `已撤销 · ${message.reason}`;
    existing.timestamp = new Date();
  } else {
    addOrderEvent({ ...message, side: "—", price: null, size: null }, `已撤销 · ${message.reason}`);
    return;
  }
  renderOrders();
}

function renderOrders() {
  const body = byId("orders-body");
  body.replaceChildren();
  if (!state.orders.length) {
    body.append(emptyRow(5, "暂无订单事件"));
    return;
  }

  for (const order of state.orders) {
    const side = `${order.outcome || "—"} ${order.side || "—"}`;
    const priceSize = order.price === null || order.price === undefined
      ? "—"
      : `${formatNumber(order.price, 4)} × ${formatNumber(order.size, 2)}`;
    const statusClass = order.status.startsWith("已成交")
      ? "warning"
      : order.status.startsWith("挂单")
        ? "positive"
        : "";
    const row = document.createElement("tr");
    row.append(
      textCell(formatClock(order.timestamp), "mono"),
      textCell(order.slug || order.market_id || "—"),
      textCell(side, "mono"),
      textCell(priceSize, "mono"),
      textCell(order.status, statusClass),
    );
    body.append(row);
  }
}

function renderBlacklist() {
  const container = byId("blacklist-items");
  container.replaceChildren();
  byId("clear-blacklist").disabled = !state.blacklist.length;

  if (!state.blacklist.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state small";
    empty.textContent = "黑名单为空";
    container.append(empty);
    return;
  }

  for (const market of state.blacklist) {
    const item = document.createElement("div");
    item.className = "blacklist-item";
    const details = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = market.question || market.slug || "Unnamed market";
    const id = document.createElement("small");
    id.textContent = market.condition_id;
    details.append(title, id);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "inline-button";
    remove.textContent = "移除";
    remove.addEventListener("click", () => {
      send({ type: "blacklist_remove", condition_id: market.condition_id });
    });
    item.append(details, remove);
    container.append(item);
  }
}

function startFarm(event) {
  event.preventDefault();
  const form = byId("farm-form");
  if (!form.reportValidity()) return;

  const validationError = validateCrossFields();
  if (validationError) {
    showToast(validationError, "error");
    return;
  }

  const payload = buildFarmPayload();
  const bankroll = Number(payload.bankroll);
  if (state.walletBalance === null || state.walletBalance <= 0) {
    showToast("钱包余额不可用或为 0，不能启动", "error");
    return;
  }
  if (bankroll > state.walletBalance) {
    showToast("Bankroll 不能高于钱包余额", "error");
    return;
  }
  if (!byId("risk-confirm").checked) {
    showToast("请先确认实盘风险", "error");
    return;
  }

  const exposure = bankroll * payload.max_concurrent_positions;
  const confirmed = window.confirm(
    [
      "即将启动真实交易：",
      `Bankroll：${formatMoney(bankroll)}`,
      `最大会话亏损：${formatMoney(payload.max_session_loss)}`,
      `同时市场：${payload.max_concurrent_positions}`,
      `理论最大并行敞口：${formatMoney(exposure)}`,
      "",
      "确认继续吗？",
    ].join("\n"),
  );
  if (!confirmed) return;

  if (send(payload)) {
    setPhase("starting");
    appendLog("Farm", "已发送 farm_create，等待后端确认", "warning");
  }
}

function stopFarm() {
  if (!["starting", "running"].includes(state.phase)) return;
  if (!window.confirm("停止 Farm 并撤销当前挂单？成交持仓仍可能需要退出处理。")) return;
  if (send({ type: "farm_cancel" })) {
    setPhase("stopping");
    appendLog("Farm", "已发送停止请求", "warning");
  }
}

function buildFarmPayload() {
  return {
    type: "farm_create",
    bankroll: byId("bankroll").value,
    max_session_loss: byId("max-session-loss").value,
    quote_depth: byId("quote-depth").value,
    max_concurrent_positions: Number.parseInt(byId("max-positions").value, 10),
    size_tiers: [],
    filters: {
      vol_min: byId("vol-min").value,
      vol_max: byId("vol-max").value,
      liq_min: byId("liq-min").value,
      liq_max: byId("liq-max").value,
      spread_min: byId("spread-min").value,
      spread_max: byId("spread-max").value,
      reward_min: byId("reward-min").value,
      time_remaining: byId("time-remaining").value,
      created_date: byId("created-date").value,
      change_24h: byId("change-24h").value,
      range_24h: byId("range-24h").value,
      zone_liq_max: optionalValue("zone-liq-max"),
      price_min: optionalValue("price-min"),
      price_max: optionalValue("price-max"),
      max_fill_loss: optionalValue("max-fill-loss"),
      min_bid_depth_mult: optionalValue("bid-depth"),
    },
  };
}

function optionalValue(id) {
  const value = byId(id).value.trim();
  return value === "" ? null : value;
}

function validateCrossFields() {
  const ranges = [
    ["vol-min", "vol-max", "24h 成交量"],
    ["liq-min", "liq-max", "流动性"],
    ["spread-min", "spread-max", "Spread"],
  ];
  for (const [minimumId, maximumId, label] of ranges) {
    if (Number(byId(minimumId).value) > Number(byId(maximumId).value)) {
      return `${label}最小值不能大于最大值`;
    }
  }
  const priceMin = optionalValue("price-min");
  const priceMax = optionalValue("price-max");
  if (priceMin !== null && priceMax !== null && Number(priceMin) > Number(priceMax)) {
    return "价格最小值不能大于最大值";
  }
  return null;
}

function updateStartAvailability() {
  const start = byId("start-button");
  const blocker = byId("start-blocker");
  let reason = "";

  if (!state.authenticated) reason = "尚未连接后端。";
  else if (state.phase === "killed") reason = "Kill Switch 已触发，请检查后重新连接。";
  else if (state.phase !== "idle") reason = "Farm 当前不处于待机状态。";
  else if (!currentWallet()) reason = "没有注册执行钱包。";
  else if (state.walletBalance === null) reason = "余额读取失败；请检查 Polygon RPC。";
  else if (state.walletBalance <= 0) reason = "钱包余额为 0，不能启动。";
  else if (!byId("farm-form").checkValidity()) reason = "请填写所有必填参数。";
  else if (!byId("risk-confirm").checked) reason = "请确认实盘风险。";

  start.disabled = Boolean(reason);
  blocker.textContent = reason || "参数就绪；点击 Start 后仍会进行最终确认。";
}

function updateExposure() {
  const bankroll = Number(byId("bankroll").value);
  const positions = Number(byId("max-positions").value);
  byId("exposure-estimate").textContent = Number.isFinite(bankroll) && Number.isFinite(positions)
    ? formatMoney(bankroll * positions)
    : "—";
  updateStartAvailability();
}

function appendLog(type, message, level = "info") {
  const log = byId("event-log");
  const entry = document.createElement("div");
  entry.className = `log-entry ${level}`;
  const time = document.createElement("span");
  time.className = "log-time";
  time.textContent = formatClock(new Date());
  const kind = document.createElement("span");
  kind.className = "log-type";
  kind.textContent = type;
  const body = document.createElement("span");
  body.className = "log-message";
  body.textContent = message;
  entry.append(time, kind, body);
  log.append(entry);
  while (log.childElementCount > 200) log.firstElementChild.remove();
  log.scrollTop = log.scrollHeight;
}

function showToast(message, kind = "info") {
  const toast = document.createElement("div");
  toast.className = `toast ${kind}`;
  toast.textContent = message;
  byId("toast-region").append(toast);
  window.setTimeout(() => toast.remove(), 4500);
}

function textCell(value, className = "") {
  const cell = document.createElement("td");
  cell.textContent = value;
  if (className) cell.className = className;
  return cell;
}

function pnlCell(value) {
  const amount = parseAmount(value) ?? 0;
  return textCell(formatSignedMoney(amount), `mono ${amount > 0 ? "positive" : amount < 0 ? "negative" : ""}`);
}

function emptyRow(columns, text) {
  const row = document.createElement("tr");
  row.className = "empty-row";
  const cell = document.createElement("td");
  cell.colSpan = columns;
  cell.textContent = text;
  row.append(cell);
  return row;
}

function formatMoney(value) {
  const amount = parseAmount(value);
  if (amount === null) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(amount);
}

function formatSignedMoney(value) {
  const amount = parseAmount(value);
  if (amount === null) return "—";
  const sign = amount > 0 ? "+" : "";
  return `${sign}${formatMoney(amount)}`;
}

function formatNumber(value, digits = 2) {
  const amount = parseAmount(value);
  if (amount === null) return "—";
  return amount.toLocaleString("en-US", { maximumFractionDigits: digits });
}

function formatClock(value) {
  const date = value instanceof Date ? value : new Date(value);
  return date.toLocaleTimeString("zh-CN", { hour12: false });
}

function formatDuration(seconds) {
  const total = Math.max(0, Number(seconds) || 0);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = Math.floor(total % 60);
  return [hours, minutes, secs].map((part) => String(part).padStart(2, "0")).join(":");
}

byId("ws-endpoint").textContent = websocketUrl();

byId("login-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const licenseKey = byId("license-key").value.trim();
  if (!licenseKey) return;
  connect(licenseKey);
});

byId("toggle-license").addEventListener("click", () => {
  const input = byId("license-key");
  const showing = input.type === "text";
  input.type = showing ? "password" : "text";
  byId("toggle-license").textContent = showing ? "显示" : "隐藏";
});

byId("disconnect-button").addEventListener("click", disconnect);
byId("farm-form").addEventListener("submit", startFarm);
byId("stop-button").addEventListener("click", stopFarm);
byId("risk-confirm").addEventListener("change", updateStartAvailability);
byId("bankroll").addEventListener("input", updateExposure);
byId("max-positions").addEventListener("input", updateExposure);
byId("farm-form").addEventListener("input", updateStartAvailability);
byId("farm-form").addEventListener("change", updateStartAvailability);

byId("refresh-wallet").addEventListener("click", () => {
  send({ type: "wallet_list" });
});

byId("copy-wallet").addEventListener("click", async () => {
  const wallet = currentWallet();
  if (!wallet) return;
  try {
    await navigator.clipboard.writeText(wallet.proxy_address);
    showToast("钱包地址已复制", "success");
  } catch (_error) {
    showToast("无法访问剪贴板", "error");
  }
});

byId("blacklist-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = byId("blacklist-url");
  if (!input.reportValidity()) return;
  if (send({ type: "blacklist_add", market_url: input.value.trim() })) input.value = "";
});

byId("clear-blacklist").addEventListener("click", () => {
  if (!state.blacklist.length) return;
  if (window.confirm(`清除全部 ${state.blacklist.length} 个手动黑名单市场？`)) {
    send({ type: "blacklist_clear" });
  }
});

byId("clear-orders").addEventListener("click", () => {
  state.orders = [];
  renderOrders();
});

byId("clear-log").addEventListener("click", () => {
  byId("event-log").replaceChildren();
});

window.addEventListener("beforeunload", (event) => {
  if (["starting", "running", "stopping"].includes(state.phase)) {
    event.preventDefault();
    event.returnValue = "";
  }
});

setConnection("offline", "未连接");
setPhase("idle");
renderPositions();
renderOrders();
renderBlacklist();
updateExposure();
document.documentElement.dataset.appReady = "true";
