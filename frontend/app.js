const { createApp, ref, reactive, computed, onMounted, nextTick, watch } = Vue;

const API = {
  async request(path, options = {}) {
    const response = await fetch(path, {
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
      ...options,
    });
    const body = response.status === 204 ? {} : await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || body.message || `请求失败 (${response.status})`);
    return body;
  },
};

createApp({
  setup() {
    const user = ref(null);
    const activeView = ref('chat');
    const authMode = ref('login');
    const busy = ref(false);
    const toast = ref('');
    const error = ref('');
    const status = ref({ ready: true, needs_first_admin: false, database_error: '' });
    const models = ref([]);
    const modelId = ref(localStorage.getItem('forest-model') || '');
    const messages = ref([]);
    const chatInput = ref('');
    const streamStatus = ref('');
    const summary = ref('');
    const chatMap = ref(null);
    const knowledgeFiles = ref([]);
    const users = ref([]);
    const readerSummaries = ref([]);
    const summariesLoading = ref(false);
    const expandedSummaryId = ref(null);
    const expandedEvalId = ref(null);
    const reportData = reactive({ overall: null, readers: [], records: [] });
    const reportsLoading = ref(false);
    const expandedReaderReportId = ref(null);
    const mapData = ref(null);
    const mapError = ref('');
    const mapReady = ref(false);
    const routeCandidates = ref([]);
    const routeResult = ref(null);
    const walkingPick = ref(null);
    const routeForm = reactive({ goals: ['轻松', '自然教育'], minutes: 90, mobility: '常规步行' });
    const loginForm = reactive({ username: '', password: '', display_name: '', confirmation: '' });
    const adminForm = reactive({ username: '', password: '', display_name: '', role: 'reader' });
    const selectedRouteId = ref('');
    let toastTimer;

    const activeModel = computed(() => models.value.find(item => item.id === modelId.value));
    const selectedRoute = computed(() => routeCandidates.value.find(item => item.id === selectedRouteId.value) || routeCandidates.value[0] || null);
    const pageHeading = computed(() => ({
      routes: ['疗养路线', '按本次状态与偏好安排分区顺序，在地图上查看完整行程。'],
      chat: ['森林咨询', '关于森林环境、健康常识、生活方式与基地路线，在这里连续交流。'],
      knowledge: ['知识资料', '本地健康科普与基地资料索引。'],
      admin: ['账号管理', '维护平台账号与访问状态。'],
    }[activeView.value] || ['森语', '森林康养智能咨询']))
    const overallPassRate = computed(() => {
      const overall = reportData.overall;
      if (!overall || !overall.total) return '—';
      return `${Math.round((overall.passed || 0) / overall.total * 100)}%`;
    });
    const avgQualityText = computed(() => {
      const overall = reportData.overall;
      if (!overall || overall.avg_quality == null) return '—';
      return `${(overall.avg_quality * 100).toFixed(0)}%`;
    });

    const cleanMarkdown = text => {
      if (!text) return '';
      return String(text)
        .replace(/\*\*(.+?)\*\*/g, '$1')
        .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, '$1$2')
        .replace(/^#{1,6}\s+/gm, '')
        .replace(/`{1,3}([^`\n]+?)`{1,3}/g, '$1')
        .replace(/^\s*>\s?/gm, '')
        .replace(/^\s*[-*+]\s+/gm, '· ')
        .replace(/^\s*(\d+)[.)]\s+/gm, '$1. ')
        .replace(/^\s*[-=]{3,}\s*$/gm, '')
        .trim();
    };
    const rate = (passed, total) => (total ? `${Math.round(passed / total * 100)}%` : '—');

    const showToast = message => {
      toast.value = message;
      clearTimeout(toastTimer);
      toastTimer = setTimeout(() => { toast.value = ''; }, 3000);
    };
    const reportError = exception => { error.value = exception.message || String(exception); };
    const clearError = () => { error.value = ''; };

    async function loadSession() {
      try {
        const data = await API.request('/api/auth/me');
        user.value = data.user;
        messages.value = data.messages || [];
        summary.value = data.summary || '';
        await loadModels();
        if (user.value.role !== 'admin' && summaryNeedsRewrite(summary.value)) {
          void refreshLegacySummary();
        }
        if (user.value.role === 'admin') {
          activeView.value = 'admin';
          await Promise.all([loadUsers(), loadSummaries(), loadReports()]);
        }
      } catch {
        user.value = null;
      }
    }

    async function loadModels() {
      const data = await API.request('/api/models');
      models.value = data.models || [];
      if (!models.value.some(item => item.id === modelId.value)) {
        modelId.value = models.value[0]?.id || '';
      }
    }

    function summaryNeedsRewrite(text) {
      return /(?:^|\n)\s*(?:[-*•]\s+|\d+[.)]\s+)|待确认|待确定|待办|尚未提及|尚未决定|尚未明确|后续交流|进一步了解|需要后续追问/.test(text || '');
    }

    async function refreshLegacySummary() {
      if (!modelId.value) return;
      try {
        const data = await API.request('/api/chat/summary/refresh', {
          method: 'POST',
          body: JSON.stringify({ model_id: modelId.value }),
        });
        summary.value = data.summary || summary.value;
      } catch {
        // Keep the existing summary visible if background migration is unavailable.
      }
    }

    async function submitAuth() {
      clearError();
      busy.value = true;
      try {
        if (authMode.value === 'register') {
          if (loginForm.password !== loginForm.confirmation) throw new Error('两次输入的密码不一致。');
          await API.request('/api/auth/register', { method: 'POST', body: JSON.stringify(loginForm) });
          authMode.value = 'login';
          showToast('注册成功，请登录。');
        } else if (authMode.value === 'first-admin') {
          if (loginForm.password !== loginForm.confirmation) throw new Error('两次输入的密码不一致。');
          await API.request('/api/auth/first-admin', { method: 'POST', body: JSON.stringify(loginForm) });
          authMode.value = 'login';
          showToast('管理员已创建，请登录。');
        } else {
          const data = await API.request('/api/auth/login', {
            method: 'POST', body: JSON.stringify({ username: loginForm.username, password: loginForm.password }),
          });
          user.value = data.user;
          messages.value = [];
          await loadSession();
          if (user.value.role !== 'admin') {
            activeView.value = 'chat';
            if (!mapData.value) await loadMapData();
          }
        }
      } catch (exception) { reportError(exception); }
      finally { busy.value = false; }
    }

    async function signOut() {
      await API.request('/api/auth/logout', { method: 'POST' });
      user.value = null;
      messages.value = [];
      activeView.value = 'chat';
    }

    async function requireLoginOnLoad() {
      // 每次页面加载（含刷新）都结束上一次会话，必须重新登录。
      try { await API.request('/api/auth/logout', { method: 'POST' }); } catch { /* 本就没有会话时忽略 */ }
      user.value = null;
      messages.value = [];
      summary.value = '';
      models.value = [];
      activeView.value = 'chat';
      authMode.value = 'login';
    }

    async function sendMessage() {
      const text = chatInput.value.trim();
      if (!text || busy.value || !modelId.value) return;
      clearError();
      messages.value.push({ role: 'user', content: text });
      const history = messages.value.map(({ role, content }) => ({ role, content }));
      messages.value.push({ role: 'assistant', content: '', tools: [], eval: null });
      const assistantMessage = messages.value[messages.value.length - 1];
      chatInput.value = '';
      busy.value = true;
      streamStatus.value = '正在连接模型…';
      chatMap.value = null;
      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          credentials: 'same-origin',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, model_id: modelId.value, history, category: 'forest' }),
        });
        if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          throw new Error(body.detail || `请求失败 (${response.status})`);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        const handleEvent = async rawEvent => {
          const dataLine = rawEvent.split(/\r?\n/).find(line => line.startsWith('data:'));
          if (!dataLine) return;
          const event = JSON.parse(dataLine.slice(5).trim());
          if (event.type === 'status') {
            streamStatus.value = event.text || '';
          } else if (event.type === 'token') {
            streamStatus.value = '';
            assistantMessage.content += event.text;
             requestAnimationFrame(() => {
               const historyElement = document.querySelector('.chat-history');
               if (historyElement) historyElement.scrollTop = historyElement.scrollHeight;
             });
          } else if (event.type === 'tool') {
            if (!assistantMessage.tools.some(item => item.name === event.name)) {
              assistantMessage.tools.push({ name: event.name, label: event.label || event.name });
            }
            streamStatus.value = `正在调用工具：${event.label || event.name}…`;
          } else if (event.type === 'eval') {
            assistantMessage.eval = event.eval;
          } else if (event.type === 'map') {
            chatMap.value = event.map;
            if (!mapData.value) mapData.value = event.map;
          } else if (event.type === 'summary') {
            summary.value = event.summary || '';
          } else if (event.type === 'error') {
            throw new Error(event.message);
          }
        };
        while (true) {
          const { value, done } = await reader.read();
          buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
          const events = buffer.split(/\r?\n\r?\n/);
          buffer = events.pop();
          for (const event of events) await handleEvent(event);
          if (done) {
            if (buffer.trim()) await handleEvent(buffer);
            break;
          }
        }
      } catch (exception) {
        streamStatus.value = '';
        assistantMessage.content += `${assistantMessage.content ? '\n\n' : ''}本轮未能完成：${exception.message}`;
      } finally { busy.value = false; streamStatus.value = ''; await nextTick(); document.querySelector('.chat-history')?.scrollTo({ top: 999999, behavior: 'smooth' }); }
    }

    async function clearConversation() {
      try {
        await API.request('/api/chat', { method: 'DELETE' });
        messages.value = [];
        summary.value = '';
        chatMap.value = null;
        showToast('对话已清除。');
      } catch (exception) { reportError(exception); }
    }

    async function loadMapData() {
      try {
        mapData.value = await API.request('/api/routes/map');
        mapReady.value = true;
        if (activeView.value === 'routes') chatMap.value = mapData.value;
      } catch (exception) { mapError.value = exception.message; }
    }

    const routePolyline = points => (points || []).map(point => point.join(',')).join(' ');

    async function calculateWalking() {
      if (!mapData.value) return;
      routeResult.value = null;
      walkingPick.value = { mode: 'start', start: '', end: '' };
      showToast('请点击地图上的起点分区。');
    }

    async function chooseMapEndpoint(zone) {
      if (!walkingPick.value) return;
      const current = { ...walkingPick.value };
      current[current.mode] = zone.id;
      if (current.mode === 'start') {
        current.mode = 'end';
        walkingPick.value = current;
        showToast(`起点：${zone.name}。请继续选择终点。`);
        return;
      }
      walkingPick.value = null;
      busy.value = true;
      clearError();
      try {
        routeResult.value = await API.request('/api/routes/walking', {
          method: 'POST', body: JSON.stringify({ start_zone: current.start, end_zone: current.end }),
        });
      } catch (exception) { reportError(exception); }
      finally { busy.value = false; }
    }

    async function recommendRoute() {
      clearError();
      routeResult.value = null;
      busy.value = true;
      try {
        const result = await API.request('/api/routes/recommend', {
          method: 'POST',
          body: JSON.stringify({ goals: routeForm.goals, duration_minutes: Number(routeForm.minutes), mobility: routeForm.mobility }),
        });
        routeCandidates.value = result.routes || [];
        selectedRouteId.value = routeCandidates.value[0]?.id || '';
        if (!routeCandidates.value.length) showToast('没有匹配线路，请调整时长或步行能力。');
      } catch (exception) { reportError(exception); }
      finally { busy.value = false; }
    }

    async function loadKnowledge() {
      try { knowledgeFiles.value = (await API.request('/api/knowledge')).files || []; }
      catch (exception) { reportError(exception); }
    }

    async function indexKnowledge() {
      busy.value = true;
      try {
        const result = await API.request('/api/knowledge/index', { method: 'POST' });
        showToast(`索引完成，新增或更新 ${result.indexed_chunks} 个文本块。`);
        await loadKnowledge();
      } catch (exception) { reportError(exception); }
      finally { busy.value = false; }
    }

    async function loadUsers() {
      try { users.value = (await API.request('/api/admin/users')).users || []; }
      catch (exception) { reportError(exception); }
    }

    async function loadSummaries() {
      summariesLoading.value = true;
      clearError();
      try { readerSummaries.value = (await API.request('/api/admin/summaries')).summaries || []; }
      catch (exception) { reportError(exception); }
      finally { summariesLoading.value = false; }
    }

    function toggleSummary(account) {
      expandedSummaryId.value = expandedSummaryId.value === account.id ? null : account.id;
    }

    function toggleEval(index) {
      expandedEvalId.value = expandedEvalId.value === index ? null : index;
    }

    async function loadReports() {
      reportsLoading.value = true;
      clearError();
      try {
        const data = await API.request('/api/admin/reports');
        reportData.overall = data.overall || null;
        reportData.readers = data.readers || [];
        reportData.records = data.records || [];
      } catch (exception) { reportError(exception); }
      finally { reportsLoading.value = false; }
    }

    function toggleReaderReport(userId) {
      expandedReaderReportId.value = expandedReaderReportId.value === userId ? null : userId;
    }

    function readerRecords(userId) {
      return reportData.records.filter(record => record.user_id === userId);
    }

    async function createUser() {
      busy.value = true;
      try {
        await API.request('/api/admin/users', { method: 'POST', body: JSON.stringify(adminForm) });
        Object.assign(adminForm, { username: '', password: '', display_name: '', role: 'reader' });
        showToast('账号已创建。'); await loadUsers();
      } catch (exception) { reportError(exception); }
      finally { busy.value = false; }
    }

    async function updateUser(target, changes) {
      try {
        await API.request(`/api/admin/users/${target.id}`, { method: 'PATCH', body: JSON.stringify(changes) });
        await loadUsers(); showToast('账号已更新。');
      } catch (exception) { reportError(exception); }
    }

    async function deleteUser(target) {
      if (!window.confirm(`确认删除账号 ${target.username} 及其对话？`)) return;
      try {
        await API.request(`/api/admin/users/${target.id}`, { method: 'DELETE' });
        await loadUsers(); showToast('账号已删除。');
      } catch (exception) { reportError(exception); }
    }

    function navigate(view) {
      if (user.value?.role === 'admin' && (view === 'routes' || view === 'chat')) view = 'admin';
      if (view === 'routes') {
        chatMap.value = mapData.value || { focus_zone: '', base_name: '', zones: [], trails: [] };
      } else if (view === 'chat' && activeView.value === 'routes') {
        chatMap.value = null;
      }
      activeView.value = view;
      clearError();
      if (view === 'routes') loadMapData();
      if (view === 'chat' && !mapData.value) loadMapData();
      if (view === 'knowledge') loadKnowledge();
      if (view === 'admin') { loadUsers(); loadSummaries(); loadReports(); }
    }

    watch(modelId, value => { if (value) localStorage.setItem('forest-model', value); });
    watch(activeView, async view => {
      if (view === 'routes' && user.value && !mapData.value) loadMapData();
    });

    onMounted(async () => {
      try { status.value = await API.request('/api/status'); }
      catch (exception) { status.value = { ready: false, database_error: exception.message }; }
      await requireLoginOnLoad();
    });

    return {
      user, activeView, authMode, busy, toast, error, status, models, modelId, activeModel,
      messages, chatInput, streamStatus, summary, chatMap, knowledgeFiles, users,
      readerSummaries, summariesLoading, expandedSummaryId, mapData, mapError, mapReady,
      routeCandidates, selectedRoute, selectedRouteId, routeForm, loginForm, adminForm,
      pageHeading, submitAuth, signOut, sendMessage, clearConversation, recommendRoute,
      calculateWalking, chooseMapEndpoint, routePolyline, loadKnowledge, indexKnowledge, createUser, updateUser, deleteUser,
      loadSummaries, toggleSummary, navigate,
      cleanMarkdown, rate, expandedEvalId, toggleEval,
      reportData, reportsLoading, expandedReaderReportId, loadReports, toggleReaderReport, readerRecords,
      overallPassRate, avgQualityText,
    };
  },
  template: `
    <div v-if="!user" class="auth-layout">
      <section class="auth-art"><div class="auth-brand">🌲 森语 · FOREST WELLNESS</div><div class="auth-title"><div class="auth-kicker">NATURE · WELLNESS · ROUTES</div><h1>把森林，<br>还给身心。</h1><p>从基地环境咨询到需求导向的步行疗养顺序，让每一次走进森林都更从容。</p></div></section>
      <section class="auth-form-side"><form class="auth-form" @submit.prevent="submitAuth"><div class="eyebrow">FOREST WELLNESS SERVICE</div><h2 style="font-size:25px;color:var(--pine);margin:10px 0 4px">进入森语</h2><p class="page-subtitle">森林康养咨询与路线规划</p><div class="auth-tabs"><button type="button" :class="{active:authMode==='login'}" @click="authMode='login'">登录</button><button type="button" :class="{active:authMode==='register'}" @click="authMode='register'">注册读者</button><button v-if="status.needs_first_admin" type="button" :class="{active:authMode==='first-admin'}" @click="authMode='first-admin'">初始化管理员</button></div><div v-if="!status.ready" class="error">数据库暂不可用：{{status.database_error}}</div><div class="field"><label class="field-label">用户名</label><input class="input" v-model="loginForm.username" autocomplete="username" required minlength="3"></div><div v-if="authMode!=='login'" class="field"><label class="field-label">称呼</label><input class="input" v-model="loginForm.display_name" autocomplete="nickname"></div><div class="field"><label class="field-label">密码</label><input class="input" v-model="loginForm.password" type="password" autocomplete="current-password" required :minlength="authMode==='login'?1:10"></div><div v-if="authMode!=='login'" class="field"><label class="field-label">确认密码</label><input class="input" v-model="loginForm.confirmation" type="password" required minlength="10"></div><div v-if="error" class="error">{{error}}</div><button class="primary" :disabled="busy||!status.ready">{{busy?'请稍候…':authMode==='login'?'登录':authMode==='register'?'创建读者账号':'创建首任管理员'}}</button><p class="sidebar-note" style="margin-top:18px">健康信息仅作一般教育；路线规划不替代医疗建议。演示版基地路线数据需以运营方核验资料为准。</p></form></section>
    </div>
    <div v-else class="shell">
      <aside class="sidebar"><div class="brand"><div class="brand-mark">🌲</div><div class="brand-copy"><div class="brand-name">森语</div><div class="brand-sub">FOREST WELLNESS</div></div></div><div class="user-block"><div class="avatar">{{(user.display_name||user.username).slice(0,1)}}</div><div class="brand-copy"><div class="user-name">{{user.display_name||user.username}}</div><div class="user-role">{{user.role==='admin'?'基地管理员':'康养访客'}}</div></div></div><div class="nav-label">工作空间</div><nav class="nav"><button v-if="user.role!=='admin'" :class="{active:activeView==='routes'}" @click="navigate('routes')"><span class="nav-icon">⌁</span><span class="nav-text">疗养路线</span></button><button v-if="user.role!=='admin'" :class="{active:activeView==='chat'}" @click="navigate('chat')"><span class="nav-icon">◌</span><span class="nav-text">森林咨询</span></button><button :class="{active:activeView==='knowledge'}" @click="navigate('knowledge')"><span class="nav-icon">▤</span><span class="nav-text">知识资料</span></button><button v-if="user.role==='admin'" :class="{active:activeView==='admin'}" @click="navigate('admin')"><span class="nav-icon">♙</span><span class="nav-text">账号管理</span></button></nav><div class="sidebar-bottom"><div v-if="user.role!=='admin'"><label class="model-label" for="model-select">对话模型</label><select id="model-select" class="model-select" v-model="modelId"><option v-for="item in models" :key="item.id" :value="item.id">{{item.label}}</option></select><div class="sidebar-note">{{activeModel?.provider||'未配置模型'}} · 支持工具调用的模型可用于路线咨询</div></div><div v-else class="sidebar-note">管理员可使用账号、知识库、摘要与测试报告管理功能</div><button class="logout" @click="signOut"><span>↪</span><span class="nav-text">退出登录</span></button></div></aside>
      <main class="main"><header class="topbar"><div class="crumb">森语　/　{{pageHeading[0]}}</div><div class="top-status"><span class="status-dot"></span>本地知识与地图服务</div></header><section class="content"><div class="page-heading"><div><div class="eyebrow">FOREST WELLNESS · FIELD GUIDE</div><h1 class="page-title">{{pageHeading[0]}}</h1><p class="page-subtitle">{{pageHeading[1]}}</p></div><span v-if="activeView==='routes'" class="pill">福州国家森林公园 · 示范版</span></div><div v-if="error" class="error">{{error}}</div>
        <template v-if="activeView==='routes'"><div class="layout-grid"><section class="panel"><h2 class="panel-title">这次想怎样走进森林？</h2><p class="panel-copy">通过几个简单问题匹配已配置的基地线路。系统不会根据健康状况进行诊断或治疗建议。</p><div class="field"><label class="field-label">你更期待的体验 <span>可多选</span></label><div class="check-grid"><label v-for="goal in ['轻松','植物','自然教育','古树','森林漫步','溪谷','登高','耐力步行','亲子']" :key="goal" class="check-chip"><input type="checkbox" :value="goal" v-model="routeForm.goals"><span>{{goal}}</span></label></div></div><div class="field"><label class="field-label">预计步行时间 <b>{{routeForm.minutes}} 分钟</b></label><div class="range-line"><input type="range" min="30" max="240" step="30" v-model="routeForm.minutes"><span class="range-value">{{routeForm.minutes}} min</span></div></div><div class="field"><label class="field-label">步行节奏</label><div class="radio-row"><label v-for="pace in ['常规步行','慢行/需休息','轮椅/行动不便']" :key="pace" class="radio-chip"><input type="radio" :value="pace" v-model="routeForm.mobility"><span>{{pace}}</span></label></div></div><button class="primary" :disabled="busy" @click="recommendRoute">{{busy?'正在整理线路…':'生成疗养顺序'}}</button><div v-if="routeCandidates.length" style="margin-top:20px"><label class="field-label">匹配线路</label><div class="route-list"><button v-for="route in routeCandidates" :key="route.id" class="route-choice" :class="{selected:selectedRoute?.id===route.id}" @click="selectedRouteId=route.id"><div class="route-choice-title">{{route.name}}</div><div class="route-meta">{{route.duration_minutes}} 分钟 · {{route.intensity}}强度 · {{route.preference_matches}} 项偏好匹配</div></button></div></div></section>
          <section class="panel"><div v-if="selectedRoute"><div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px"><div><h2 class="panel-title">分区疗养顺序</h2><p class="panel-copy">{{selectedRoute.name}} · 根据你本次填写的体验偏好排序</p></div><span class="pill">顺序示意</span></div><div class="sequence"><div v-for="(zone,index) in selectedRoute.zones" :key="zone.id" class="sequence-stop"><div class="stop-number">{{index+1}}</div><div class="stop-name">{{zone.name}}</div><div class="stop-copy">{{zone.function}}</div></div></div><div class="route-stats"><div><strong>{{selectedRoute.distance_km}} km</strong>规划距离</div><div><strong>{{selectedRoute.duration_minutes}} 分钟</strong>预计时长</div><div><strong>{{selectedRoute.intensity}}</strong>活动强度</div></div><div class="map-frame"><svg v-if="mapData" class="virtual-map" viewBox="0 0 960 620" role="img" aria-label="基地虚拟二维分区示意图"><path class="map-water" d="M40 150 C180 100 190 230 340 200 S560 100 700 180 S820 240 940 140"/><path class="map-grove" d="M70 360 C170 270 260 320 300 410 S180 560 80 490Z"/><path class="map-grove map-grove-alt" d="M590 70 C710 40 870 65 900 190 S780 270 690 220Z"/><polyline v-for="trail in mapData.trails" :key="trail.from + trail.to" :points="routePolyline(trail.points)" class="trail-line"/><polyline v-if="selectedRoute" :points="routePolyline(selectedRoute.zones.map(zone => zone.map_position))" class="suggested-line"/><polyline v-if="routeResult" :points="routePolyline(routeResult.coordinates)" class="calculated-line"/><g v-for="zone in mapData.zones" :key="zone.id" class="map-zone" :class="{ selected: selectedRoute?.zones.some(item => item.id === zone.id) }" tabindex="0" @click="chooseMapEndpoint(zone)" @keydown.enter="chooseMapEndpoint(zone)"><circle :cx="zone.map_position[0]" :cy="zone.map_position[1]" r="21"/><text :x="zone.map_position[0]" :y="zone.map_position[1] + 5" text-anchor="middle">{{selectedRoute?.zones.findIndex(item => item.id === zone.id) >= 0 ? selectedRoute.zones.findIndex(item => item.id === zone.id) + 1 : '·'}}</text><text class="zone-label" :x="zone.map_position[0]" :y="zone.map_position[1] + 43" text-anchor="middle">{{zone.name}}</text></g><text class="map-caption" x="36" y="38">{{mapData.name}}</text><text class="map-north" x="906" y="54">N ↑</text></svg><div v-else class="map-placeholder">{{mapError||'正在加载基地地图…'}}</div><div class="map-legend"><span>虚拟分区与步道关系示意，不表示真实地理位置</span><span>{{mapData?.base_name||'基地示意图'}}</span></div></div><div class="notice">{{mapData?.source_note||'分区坐标与路线为开发示意，未经过园方核验。'}} 不得据此导航；请按基地现场指引核实开放、路况及自身健康适宜性。</div><button class="primary" style="margin-top:13px" :disabled="busy" @click="calculateWalking">{{routeResult?'重新规划起终点步行路线':'规划示意步行路线'}}</button><div v-if="routeResult" class="success" style="margin-top:12px">示意步行路线：{{routeResult.distance_km.toFixed(2)}} km · 约 {{routeResult.duration_minutes}} 分钟 · {{routeResult.source}}<br>{{routeResult.instructions.join(' → ')}}</div></div><div v-else><h2 class="panel-title">路线地图</h2><p class="panel-copy">先填写偏好并生成分区疗养顺序，地图会在此处显示线路节点。</p><div class="map-frame"><svg v-if="mapData" class="virtual-map" viewBox="0 0 960 620" role="img" aria-label="基地虚拟二维分区示意图"><path class="map-water" d="M40 150 C180 100 190 230 340 200 S560 100 700 180 S820 240 940 140"/><path class="map-grove" d="M70 360 C170 270 260 320 300 410 S180 560 80 490Z"/><path class="map-grove map-grove-alt" d="M590 70 C710 40 870 65 900 190 S780 270 690 220Z"/><polyline v-for="trail in mapData.trails" :key="trail.from + trail.to" :points="routePolyline(trail.points)" class="trail-line"/><g v-for="zone in mapData.zones" :key="zone.id" class="map-zone" :class="{ selected: selectedRoute?.zones.some(item => item.id === zone.id) }" tabindex="0" @click="chooseMapEndpoint(zone)" @keydown.enter="chooseMapEndpoint(zone)"><circle :cx="zone.map_position[0]" :cy="zone.map_position[1]" r="21"/><text :x="zone.map_position[0]" :y="zone.map_position[1]+5" text-anchor="middle">{{chatMap.focus_zone===zone.id?'●':'·'}}</text><text class="zone-label" :x="zone.map_position[0]" :y="zone.map_position[1]+43" text-anchor="middle">{{zone.name}}</text></g><text class="map-caption" x="36" y="38">基地虚拟示意</text></svg><div v-else class="map-placeholder">{{mapError||'正在加载基地地图…'}}</div><div class="map-legend"><span>点位与步道为演示数据</span><span>{{chatMap.base_name}}</span></div></div></section></div></template>
        <template v-else-if="activeView==='chat'"><div class="chat-layout"><section class="panel"><h2 class="panel-title">森林咨询</h2><p class="panel-copy">森林环境、健康常识、生活方式与基地路线问题，都可以在这里连续交流。</p><div class="chat-history" style="max-height:58vh;overflow:auto"><div v-if="!messages.length" class="empty-state">欢迎提问。你可以咨询森林生态、健康常识、生活方式，也可以了解基地分区与示意路线。</div><div v-for="(message,index) in messages" :key="index" class="bubble" :class="message.role"><div class="bubble-text">{{cleanMarkdown(message.content)}}</div><div v-if="message.role==='assistant' && message.tools && message.tools.length" class="message-tools"><span v-for="tool in message.tools" :key="tool.name" class="tool-chip">工具：{{tool.label}}</span></div><div v-if="message.role==='assistant' && message.eval" class="eval-block"><button type="button" class="eval-toggle" @click="toggleEval(index)"><span class="eval-dot" :class="message.eval.overall_pass ? 'pass' : 'fail'"></span>{{message.eval.overall_pass ? '质量检查通过' : '质量检查未通过'}} · 工具 {{message.eval.tools.length}} 项 · {{expandedEvalId===index ? '收起评价' : '查看检查明细'}}</button><div v-if="expandedEvalId===index" class="eval-panel"><div class="eval-metrics"><div class="eval-metric"><strong>整体结果</strong><span :class="message.eval.overall_pass ? 'ok' : 'bad'">{{message.eval.overall_pass ? '通过' : '未通过'}}</span></div><div class="eval-metric"><strong>质量评分</strong><span>{{message.eval.quality_score != null ? (message.eval.quality_score*100).toFixed(0)+'%' : '—'}}</span></div></div><table class="eval-checks"><tbody><tr v-for="(check,ci) in message.eval.checks" :key="ci"><td class="check-name">{{check.name}}</td><td class="check-state" :class="check.state">{{check.state==='pass' ? '✓ 通过' : '✗ 未通过'}}</td><td class="check-note">{{check.note}}</td></tr></tbody></table></div></div></div><div v-if="busy" class="bubble">{{streamStatus||'正在生成回复…'}}</div></div><div v-if="chatMap" class="chat-map-card"><div class="chat-map-head"><div><strong>基地虚拟二维示意</strong><div class="chat-map-sub">由虚拟地图工具生成 · 仅用于分区关系展示</div></div><span class="pill">非导航</span></div><svg class="virtual-map chat-map" viewBox="0 0 960 620" role="img" aria-label="基地虚拟二维分区示意图"><path class="map-water" d="M40 150 C180 100 190 230 340 200 S560 100 700 180 S820 240 940 140"/><path class="map-grove" d="M70 360 C170 270 260 320 300 410 S180 560 80 490Z"/><path class="map-grove map-grove-alt" d="M590 70 C710 40 870 65 900 190 S780 270 690 220Z"/><polyline v-for="trail in chatMap.trails" :key="trail.from + trail.to" :points="routePolyline(trail.points)" class="trail-line"/><g v-for="zone in chatMap.zones" :key="zone.id" class="map-zone" :class="{selected:chatMap.focus_zone===zone.id}"><circle :cx="zone.map_position[0]" :cy="zone.map_position[1]" r="21"/><text :x="zone.map_position[0]" :y="zone.map_position[1]+5" text-anchor="middle">{{chatMap.focus_zone===zone.id?'●':'·'}}</text><text class="zone-label" :x="zone.map_position[0]" :y="zone.map_position[1]+43" text-anchor="middle">{{zone.name}}</text></g><text class="map-caption" x="36" y="38">基地虚拟示意</text></svg><div class="map-legend"><span>点位与步道为演示数据</span><span>{{chatMap.base_name}}</span></div></div><div class="chat-composer"><textarea class="textarea" v-model="chatInput" :placeholder="'输入你想了解的问题，例如：森林环境对身心健康有哪些影响？'" @keydown.ctrl.enter.prevent="sendMessage"></textarea><button class="primary" style="width:92px" :disabled="busy||!chatInput.trim()" @click="sendMessage">发送</button></div><button style="margin-top:10px;border:0;background:none;color:var(--muted);font-size:13px" @click="clearConversation">清空对话与记忆</button></section><aside class="panel"><h2 class="panel-title">你也可以问</h2><p class="panel-copy">选择示例问题，将内容填入输入框后发送。</p><button class="route-choice" style="width:100%;margin-bottom:8px" @click="chatInput='森林生态系统如何帮助调节局地气候？'">森林生态系统如何帮助调节局地气候？</button><button class="route-choice" style="width:100%;margin-bottom:8px" @click="chatInput='日常运动时如何识别需要停止活动的身体信号？'">日常运动时如何识别需要停止活动的身体信号？</button><button class="route-choice" style="width:100%;margin-bottom:8px" @click="chatInput='有哪些容易坚持的睡眠和放松习惯？'">有哪些容易坚持的睡眠和放松习惯？</button><div class="chat-memory"><h3>对话记忆摘要</h3><p v-if="summary">{{cleanMarkdown(summary)}}</p><p v-else class="memory-empty">对话积累后，这里会显示便于后续交流的简要摘要。</p></div></aside></div></template>
        <template v-else-if="activeView==='knowledge'"><section class="panel"><div style="display:flex;justify-content:space-between;align-items:center;gap:12px"><div><h2 class="panel-title">本地知识资料</h2><p class="panel-copy">健康内容仅作一般教育，不代替专业诊疗。</p></div><button v-if="user.role==='admin'" class="primary" style="width:auto" :disabled="busy" @click="indexKnowledge">{{busy?'索引中…':'索引 / 更新资料'}}</button></div><table class="table"><thead><tr><th>文件</th><th>格式</th><th>相对路径</th></tr></thead><tbody><tr v-for="file in knowledgeFiles" :key="file.path"><td>{{file.name}}</td><td>{{file.type}}</td><td>{{file.path}}</td></tr><tr v-if="!knowledgeFiles.length"><td colspan="3">暂无可用资料</td></tr></tbody></table></section></template>
        <template v-else-if="activeView==='admin'"><div class="layout-grid"><section class="panel"><h2 class="panel-title">创建账号</h2><div class="field"><label class="field-label">用户名</label><input class="input" v-model="adminForm.username"></div><div class="field"><label class="field-label">称呼</label><input class="input" v-model="adminForm.display_name"></div><div class="field"><label class="field-label">初始密码（至少 10 位）</label><input class="input" v-model="adminForm.password" type="password"></div><div class="field"><label class="field-label">角色</label><select class="select" v-model="adminForm.role"><option value="reader">读者</option><option value="admin">管理员</option></select></div><button class="primary" :disabled="busy" @click="createUser">创建账号</button></section><section class="panel"><h2 class="panel-title">账号维护</h2><table class="table"><thead><tr><th>账号</th><th>角色</th><th>状态</th><th>操作</th></tr></thead><tbody><tr v-for="account in users" :key="account.id"><td>{{account.username}}<br>{{account.display_name}}</td><td>{{account.role}}</td><td>{{account.status}}</td><td><button class="route-choice" @click="updateUser(account,{status:account.status==='active'?'disabled':'active'})">{{account.status==='active'?'停用':'启用'}}</button> <button class="route-choice" @click="updateUser(account,{role:account.role==='admin'?'reader':'admin'})">切换角色</button> <button class="route-choice" @click="deleteUser(account)">删除</button></td></tr></tbody></table></section></div><section class="panel" style="margin-top:18px"><div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px"><div><h2 class="panel-title">读者对话摘要</h2><p class="panel-copy">仅展示普通读者的滚动摘要，不展示逐条聊天记录。摘要仅用于服务跟进，请勿对外传播。</p></div><button class="route-choice" style="white-space:nowrap" :disabled="summariesLoading" @click="loadSummaries">{{summariesLoading?'加载中…':'刷新摘要'}}</button></div><table class="table"><thead><tr><th>读者</th><th>状态</th><th>摘要更新时间</th><th>操作</th></tr></thead><tbody><template v-for="account in readerSummaries" :key="account.id"><tr><td>{{account.username}}<br>{{account.display_name}}</td><td>{{account.status==='active'?'启用':'停用'}}</td><td>{{account.updated_at||'—'}}</td><td><button class="route-choice" @click="toggleSummary(account)">{{expandedSummaryId===account.id?'收起摘要':'查看摘要'}}</button></td></tr><tr v-if="expandedSummaryId===account.id"><td colspan="4"><div class="summary-block">{{cleanMarkdown(account.summary)||'该读者暂无对话摘要。'}}</div></td></tr></template><tr v-if="!readerSummaries.length"><td colspan="4">{{summariesLoading?'正在加载…':'暂无普通读者账号'}}</td></tr></tbody></table></section><section class="panel" style="margin-top:18px"><div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px"><div><h2 class="panel-title">测试报告</h2><p class="panel-copy">汇总每位读者的对话质检结果：工具调用、规则检查与模型质量评分，仅用于服务改进。</p></div><button class="route-choice" style="white-space:nowrap" :disabled="reportsLoading" @click="loadReports">{{reportsLoading?'加载中…':'刷新报告'}}</button></div><div v-if="reportData.overall" class="report-metrics"><div class="report-metric"><strong>{{reportData.overall.total||0}}</strong><span>测试轮次</span></div><div class="report-metric"><strong>{{overallPassRate}}</strong><span>通过率</span></div><div class="report-metric"><strong>{{avgQualityText}}</strong><span>平均质量分</span></div></div><table class="table"><thead><tr><th>读者</th><th>测试轮次</th><th>通过率</th><th>平均质量分</th><th>最近测试</th><th>操作</th></tr></thead><tbody><template v-for="reader in reportData.readers" :key="reader.user_id"><tr><td>{{reader.username}}<br>{{reader.display_name}}</td><td>{{reader.turns}}</td><td>{{rate(reader.passed_turns, reader.turns)}}</td><td>{{reader.avg_quality != null ? (reader.avg_quality*100).toFixed(0)+'%' : '—'}}</td><td>{{reader.last_at||'—'}}</td><td><button class="route-choice" @click="toggleReaderReport(reader.user_id)">{{expandedReaderReportId===reader.user_id?'收起明细':'查看明细'}}</button></td></tr><tr v-if="expandedReaderReportId===reader.user_id"><td colspan="6"><div v-for="rec in readerRecords(reader.user_id)" :key="rec.id" class="eval-record"><div class="eval-record-q">{{cleanMarkdown(rec.query)}}</div><div class="eval-record-meta"><span v-for="tool in rec.detail.tools" :key="tool.name" class="tool-chip">工具：{{tool.label}}</span><span class="eval-record-state" :class="rec.passed ? 'pass':'fail'">{{rec.passed ? '通过' : '未通过'}}</span><span v-if="rec.detail.quality_score != null" class="eval-record-score">质量分 {{(rec.detail.quality_score*100).toFixed(0)}}%</span></div></div><div v-if="!readerRecords(reader.user_id).length" class="memory-empty">该读者暂无测试记录。</div></td></tr></template><tr v-if="!reportData.readers.length"><td colspan="6">{{reportsLoading?'正在加载…':'暂无测试报告'}}</td></tr></tbody></table></section></template>
      </section></main><div v-if="toast" class="toast">{{toast}}</div>
    </div>
  `,
}).mount('#app');
