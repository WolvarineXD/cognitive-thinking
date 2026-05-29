/**
 * Cognitive AI — ChatGPT-style Frontend
 * Persistent conversations, streaming responses, memory proof
 */

// ── State ──────────────────────────────────────────
let currentConvId = null;
let isStreaming = false;
let conversations = [];

// ── DOM ────────────────────────────────────────────
const $ = id => document.getElementById(id);
const convList = $('conv-list');
const messagesEl = $('messages');
const inputEl = $('input');
const sendBtn = $('send-btn');
const chatView = $('chat-view');
const emptyState = $('empty-state');
const chatTitle = $('chat-title');

// ── Init ───────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    setupInput();
    await loadConversations();
    await loadMemoryBreakdown();
});

function setupInput() {
    inputEl.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    inputEl.addEventListener('input', () => {
        inputEl.style.height = 'auto';
        inputEl.style.height = Math.min(inputEl.scrollHeight, 150) + 'px';
    });
}

// ── Conversations CRUD ─────────────────────────────

async function loadConversations() {
    try {
        const res = await fetch('/api/conversations');
        conversations = await res.json();
        renderConvList();
    } catch (e) {
        console.warn('Failed to load conversations:', e);
    }
}

function renderConvList() {
    convList.innerHTML = '';
    if (conversations.length === 0) {
        convList.innerHTML = '<div style="padding:20px;text-align:center;color:var(--text-3);font-size:12px">No conversations yet.<br>Start a new chat!</div>';
        return;
    }
    for (const c of conversations) {
        const div = document.createElement('div');
        div.className = `conv-item${c.id === currentConvId ? ' active' : ''}`;
        div.onclick = () => openConversation(c.id);
        div.innerHTML = `
            <span class="conv-title">${escapeHtml(c.title)}</span>
            <button class="conv-delete" onclick="event.stopPropagation();deleteConversation('${c.id}')" title="Delete">&times;</button>
        `;
        convList.appendChild(div);
    }
}

async function createConversation() {
    try {
        const res = await fetch('/api/conversations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: 'New Chat' })
        });
        const conv = await res.json();
        await loadConversations();
        await openConversation(conv.id);
    } catch (e) {
        console.error('Create conv failed:', e);
    }
}

async function openConversation(convId) {
    currentConvId = convId;
    try {
        const res = await fetch(`/api/conversations/${convId}`);
        const conv = await res.json();

        // Switch to chat view
        emptyState.style.display = 'none';
        chatView.style.display = 'flex';
        chatTitle.textContent = conv.title;

        // Render messages
        messagesEl.innerHTML = '';
        for (const msg of conv.messages) {
            if (msg.role === 'user') {
                appendUserMessage(msg.content, msg.metadata);
            } else {
                appendAIMessage(msg.content, msg.metadata);
            }
        }
        scrollToBottom();

        // Highlight in sidebar
        renderConvList();
        inputEl.focus();
    } catch (e) {
        console.error('Open conv failed:', e);
    }
}

async function deleteConversation(convId) {
    if (!confirm('Delete this conversation?')) return;
    try {
        await fetch(`/api/conversations/${convId}`, { method: 'DELETE' });
        if (currentConvId === convId) {
            currentConvId = null;
            emptyState.style.display = '';
            chatView.style.display = 'none';
        }
        await loadConversations();
    } catch (e) {
        console.error('Delete failed:', e);
    }
}

// ── Send Message ───────────────────────────────────

async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text || isStreaming) return;

    // Auto-create conversation if needed
    if (!currentConvId) {
        const res = await fetch('/api/conversations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: 'New Chat' })
        });
        const conv = await res.json();
        currentConvId = conv.id;
        emptyState.style.display = 'none';
        chatView.style.display = 'flex';
        chatTitle.textContent = 'New Chat';
        messagesEl.innerHTML = '';
        await loadConversations();
    }

    // Show user message
    appendUserMessage(text);
    inputEl.value = '';
    inputEl.style.height = 'auto';
    scrollToBottom();

    // Stream response
    isStreaming = true;
    sendBtn.disabled = true;
    showTyping();

    try {
        await streamChat(text);
    } catch (e) {
        hideTyping();
        appendAIMessage('Connection error. Is the server running?', {});
        console.error('Chat error:', e);
    }

    isStreaming = false;
    sendBtn.disabled = false;
    inputEl.focus();
    await loadMemoryBreakdown();
}

async function quickChat(text) {
    inputEl.value = text;
    await sendMessage();
}

// ── Streaming ──────────────────────────────────────

async function streamChat(message) {
    const res = await fetch(`/api/conversations/${currentConvId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, top_k: 5 }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let aiRow = null;
    let aiContent = '';
    let metadata = null;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            try {
                const data = JSON.parse(line.slice(6));

                if (data.type === 'metadata') {
                    hideTyping();
                    metadata = data;
                    aiRow = appendAIMessageStreaming();

                    // Update the user message with classification info
                    updateLastUserMsgMeta(data);
                }

                if (data.type === 'token' && aiRow) {
                    aiContent += data.content;
                    aiRow.querySelector('.msg-content').textContent = aiContent;
                    scrollToBottom();
                }

                if (data.type === 'title') {
                    chatTitle.textContent = data.title;
                    loadConversations();
                }

                if (data.type === 'done') {
                    // Finalize: format + add memory proof
                    if (aiRow) {
                        const contentEl = aiRow.querySelector('.msg-content');
                        contentEl.innerHTML = formatMarkdown(aiContent);
                        if (metadata && metadata.memories_used && metadata.memories_used.length > 0) {
                            const proof = buildMemoryProof(metadata);
                            aiRow.querySelector('.msg-body').appendChild(proof);
                        }
                    }
                    loadConversations();
                }
            } catch (e) {}
        }
    }
}

// ── Update user message with memory classification ─

function updateLastUserMsgMeta(meta) {
    const userRows = messagesEl.querySelectorAll('.msg-row.user');
    if (!userRows.length) return;
    const lastRow = userRows[userRows.length - 1];

    // Don't add twice
    if (lastRow.querySelector('.user-mem-type')) return;

    const memType = meta.memory_type || 'episodic';
    const dotClass = memType + '-dot';

    const badge = document.createElement('div');
    badge.className = 'user-mem-type';
    badge.innerHTML = `<span class="mem-dot ${dotClass}"></span> stored as <strong style="color:var(--text-2);margin:0 2px">${memType}</strong> memory`;

    // Add classification scores if available
    if (meta.classification_scores) {
        const scores = meta.classification_scores;
        let pillsHtml = ' &middot; ';
        for (const [type, score] of Object.entries(scores)) {
            if (score > 0) {
                const isActive = type === memType ? `active ${type}` : '';
                pillsHtml += `<span class="class-score-pill ${isActive}">${type}: ${(score * 100).toFixed(0)}%</span> `;
            }
        }
        badge.innerHTML += pillsHtml;
    }

    lastRow.querySelector('.msg-body').appendChild(badge);
}

// ── Message Rendering ──────────────────────────────

function appendUserMessage(content, metadata) {
    const row = document.createElement('div');
    row.className = 'msg-row user';
    row.innerHTML = `
        <div class="msg-avatar">U</div>
        <div class="msg-body">
            <div class="msg-content">${escapeHtml(content)}</div>
        </div>
    `;

    // Show saved memory type from persisted metadata
    if (metadata && metadata.memory_type) {
        const memType = metadata.memory_type;
        const dotClass = memType + '-dot';
        const badge = document.createElement('div');
        badge.className = 'user-mem-type';
        badge.innerHTML = `<span class="mem-dot ${dotClass}"></span> stored as <strong style="color:var(--text-2);margin:0 2px">${memType}</strong>`;
        if (metadata.classification_scores) {
            let pillsHtml = ' &middot; ';
            for (const [type, score] of Object.entries(metadata.classification_scores)) {
                if (score > 0) {
                    const isActive = type === memType ? `active ${type}` : '';
                    pillsHtml += `<span class="class-score-pill ${isActive}">${type}: ${(score * 100).toFixed(0)}%</span> `;
                }
            }
            badge.innerHTML += pillsHtml;
        }
        row.querySelector('.msg-body').appendChild(badge);
    }

    messagesEl.appendChild(row);
}

function appendAIMessage(content, metadata) {
    const row = document.createElement('div');
    row.className = 'msg-row assistant';
    row.innerHTML = `
        <div class="msg-avatar">&#x2726;</div>
        <div class="msg-body">
            <div class="msg-content">${formatMarkdown(content)}</div>
        </div>
    `;
    // Add memory proof if available
    if (metadata && metadata.memories_used && metadata.memories_used.length > 0) {
        row.querySelector('.msg-body').appendChild(buildMemoryProof(metadata));
    }
    messagesEl.appendChild(row);
}

function appendAIMessageStreaming() {
    const row = document.createElement('div');
    row.className = 'msg-row assistant';
    row.innerHTML = `
        <div class="msg-avatar">&#x2726;</div>
        <div class="msg-body">
            <div class="msg-content"><span style="color:var(--text-3)">Thinking...</span></div>
        </div>
    `;
    messagesEl.appendChild(row);
    return row;
}

// ── Memory Proof Panel ─────────────────────────────

function buildMemoryProof(meta) {
    const container = document.createElement('div');
    container.className = 'memory-proof';

    const memCount = meta.memories_used ? meta.memories_used.length : 0;
    const toggleId = 'proof-' + Date.now();

    // Count memory types used
    const typeCounts = {};
    if (meta.memories_used) {
        for (const m of meta.memories_used) {
            typeCounts[m.type] = (typeCounts[m.type] || 0) + 1;
        }
    }
    const typeChips = Object.entries(typeCounts)
        .map(([t, c]) => `<span class="proof-type ${t}" style="margin-left:4px">${c} ${t}</span>`)
        .join('');

    container.innerHTML = `
        <button class="memory-proof-toggle" onclick="toggleProof('${toggleId}', this)">
            <span class="arrow">&#x25B6;</span>
            ${memCount} memories retrieved${meta.retrieval_time_ms ? ` &middot; ${meta.retrieval_time_ms}ms` : ''}
            ${typeChips}
        </button>
        <div class="memory-proof-body" id="${toggleId}">
            ${buildProofMeta(meta)}
            ${buildProofMemories(meta.memories_used || [])}
        </div>
    `;
    return container;
}

function buildProofMeta(meta) {
    let html = '<div class="proof-meta">';
    if (meta.retrieval_time_ms) html += `<span class="proof-tag time">${meta.retrieval_time_ms}ms</span>`;
    if (meta.total_candidates) html += `<span class="proof-tag count">${meta.total_candidates} candidates</span>`;
    if (meta.intent) html += `<span class="proof-tag intent">${meta.intent}</span>`;
    if (meta.entities_extracted) {
        for (const e of meta.entities_extracted.slice(0, 4)) {
            html += `<span class="proof-tag" style="color:var(--text-2);border-color:var(--border)">${escapeHtml(e)}</span>`;
        }
    }
    html += '</div>';
    return html;
}

function buildProofMemories(memories) {
    if (!memories.length) return '<div style="color:var(--text-3);font-size:12px">No relevant memories found</div>';
    let html = '';
    for (const m of memories) {
        const semW = Math.max(3, m.semantic * 100);
        const recW = Math.max(3, m.recency * 100);
        const impW = Math.max(3, m.importance * 100);
        const gphW = Math.max(3, m.graph * 100);
        html += `
        <div class="proof-memory">
            <div class="proof-memory-header">
                <span class="proof-type ${m.type}">${m.type}</span>
                <span class="proof-score">${m.score.toFixed(3)}</span>
            </div>
            <div class="proof-text">${escapeHtml(m.content)}</div>
            <div class="score-bars" title="Semantic: ${m.semantic} | Recency: ${m.recency} | Importance: ${m.importance} | Graph: ${m.graph}">
                <div class="bar-semantic" style="width:${semW}%"></div>
                <div class="bar-recency" style="width:${recW}%"></div>
                <div class="bar-importance" style="width:${impW}%"></div>
                <div class="bar-graph" style="width:${gphW}%"></div>
            </div>
            <div class="score-legend">
                <span class="score-legend-item"><span class="score-legend-dot" style="background:#818cf8"></span>Sem ${m.semantic.toFixed(2)}</span>
                <span class="score-legend-item"><span class="score-legend-dot" style="background:#34d399"></span>Rec ${m.recency.toFixed(2)}</span>
                <span class="score-legend-item"><span class="score-legend-dot" style="background:#fbbf24"></span>Imp ${m.importance.toFixed(2)}</span>
                <span class="score-legend-item"><span class="score-legend-dot" style="background:#ec4899"></span>Graph ${m.graph.toFixed(2)}</span>
            </div>
        </div>`;
    }
    return html;
}

function toggleProof(id, btn) {
    const body = document.getElementById(id);
    const arrow = btn.querySelector('.arrow');
    if (body.classList.contains('open')) {
        body.classList.remove('open');
        arrow.classList.remove('open');
    } else {
        body.classList.add('open');
        arrow.classList.add('open');
    }
}

// ── Typing Indicator ───────────────────────────────

function showTyping() {
    let el = document.querySelector('.typing-row');
    if (!el) {
        el = document.createElement('div');
        el.className = 'typing-row';
        el.innerHTML = `
            <div class="msg-avatar" style="background:linear-gradient(135deg,#1e1b4b,#312e81);color:var(--accent)">&#x2726;</div>
            <div class="typing-dots"><span></span><span></span><span></span></div>
        `;
        messagesEl.appendChild(el);
    }
    el.classList.add('active');
    scrollToBottom();
}

function hideTyping() {
    const el = document.querySelector('.typing-row');
    if (el) el.remove();
}

// ── Memory Breakdown (Sidebar) ─────────────────────

async function loadMemoryBreakdown() {
    try {
        const res = await fetch('/api/memory-breakdown');
        const data = await res.json();
        const b = data.breakdown;
        const total = data.total;

        // Update counts
        $('mem-total').textContent = total;
        $('mc-episodic').textContent = b.episodic.count;
        $('mc-semantic').textContent = b.semantic.count;
        $('mc-procedural').textContent = b.procedural.count;
        $('mc-working').textContent = b.working.count;

        // Update stacked bar
        if (total > 0) {
            $('bar-episodic').style.width = (b.episodic.count / total * 100) + '%';
            $('bar-semantic').style.width = (b.semantic.count / total * 100) + '%';
            $('bar-procedural').style.width = (b.procedural.count / total * 100) + '%';
            $('bar-working').style.width = (b.working.count / total * 100) + '%';
        } else {
            $('bar-episodic').style.width = '0%';
            $('bar-semantic').style.width = '0%';
            $('bar-procedural').style.width = '0%';
            $('bar-working').style.width = '0%';
        }

        // Store for preview expansion
        window._memBreakdown = b;

        // Update graph stats
        const statsRes = await fetch('/api/stats');
        const stats = await statsRes.json();
        $('s-graph').textContent = stats.graph_stats?.nodes || 0;
        $('s-edges').textContent = stats.graph_stats?.edges || 0;
    } catch (e) {
        console.error('Failed to load memory breakdown:', e);
    }
}

function toggleMemPreview(el) {
    const type = el.dataset.type;
    const previewEl = $('mp-' + type);
    if (previewEl.classList.contains('open')) {
        previewEl.classList.remove('open');
        return;
    }
    // Close others
    document.querySelectorAll('.mem-preview.open').forEach(p => p.classList.remove('open'));

    // Populate
    const b = window._memBreakdown;
    if (!b || !b[type]) return;

    const recent = b[type].recent || [];
    if (recent.length === 0) {
        previewEl.innerHTML = '<div class="mem-preview-item" style="font-style:italic">No memories yet</div>';
    } else {
        previewEl.innerHTML = recent.map(m =>
            `<div class="mem-preview-item" title="${escapeHtml(m.content)}">${escapeHtml(m.content)}</div>`
        ).join('');
    }
    previewEl.classList.add('open');
}

// ── Reflection ─────────────────────────────────────

async function runReflection() {
    try {
        const res = await fetch('/api/reflection', { method: 'POST' });
        const r = await res.json();
        alert(`Reflection: Reviewed ${r.memories_reviewed} memories, merged ${r.duplicates_merged}, pruned ${r.memories_pruned}`);
        loadMemoryBreakdown();
    } catch (e) {
        alert('Reflection failed: ' + e.message);
    }
}

// ── Sidebar Toggle ─────────────────────────────────

function toggleSidebar() {
    $('sidebar').classList.toggle('hidden');
}

// ── Utilities ──────────────────────────────────────

function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

function formatMarkdown(text) {
    let h = escapeHtml(text);
    h = h.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    h = h.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
    h = h.replace(/\n/g, '<br>');
    return h;
}
