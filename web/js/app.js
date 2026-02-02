const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const chatContainer = document.getElementById('chat-container');

// CONFIRMATION OF UPDATE
console.log("APP JS V3 LOADED - CHATBOT MODE");

// ============================================================================
// SESSION & CONVERSATION TRACKING
// ============================================================================
let sessionId = null;  // Will be set by server on first message
let conversationTurns = 0;

// ============================================================================
// THEME TOGGLE
// ============================================================================
function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('sentra-theme', newTheme);
    updateThemeButton(newTheme);
    console.log('[Theme] Switched to:', newTheme);
}

function updateThemeButton(theme) {
    const sunIcon = document.getElementById('sun-icon');
    const moonIcon = document.getElementById('moon-icon');
    const themeLabel = document.getElementById('theme-label');

    if (theme === 'dark') {
        sunIcon?.classList.remove('hidden');
        moonIcon?.classList.add('hidden');
        if (themeLabel) themeLabel.textContent = 'Light';
    } else {
        sunIcon?.classList.add('hidden');
        moonIcon?.classList.remove('hidden');
        if (themeLabel) themeLabel.textContent = 'Dark';
    }
}

// Initialize theme on load
(function initTheme() {
    const savedTheme = localStorage.getItem('sentra-theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => updateThemeButton(savedTheme));
    } else {
        updateThemeButton(savedTheme);
    }
})();

// ============================================================================
// CHAT SUBMIT HANDLER
// ============================================================================
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = userInput.value.trim();
    if (!message) return;

    // Add User Message to UI
    appendMessage(message, 'user');
    userInput.value = '';

    // Show Loading
    const loadingId = appendLoading();

    const isReduceMode = document.getElementById('mode-toggle').checked;
    const mode = isReduceMode ? 'reduce_hallucination' : 'default';

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                mode,
                session_id: sessionId  // Send session_id for continuity
            })
        });

        const data = await response.json();
        document.getElementById(loadingId).remove();

        // Update session_id from server response
        if (data.session_id) {
            sessionId = data.session_id;
            console.log("[Session] Active session:", sessionId);
        }

        // Track conversation turns
        conversationTurns++;
        updateConversationIndicator();

        // Add Bot Message
        appendMessage(marked.parse(data.answer), 'bot');

        // Update Sidebar
        updateSidebar(data);

    } catch (error) {
        console.error(error);
        document.getElementById(loadingId).remove();
        appendMessage("Sorry, connection error. Please try again.", 'bot');
    }
});

// ============================================================================
// NEW CHAT FUNCTION
// ============================================================================
async function startNewChat() {
    // Clear session on server if exists
    if (sessionId) {
        try {
            await fetch('/api/chat/clear', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId })
            });
            console.log("[Session] Cleared:", sessionId);
        } catch (e) {
            console.warn("Failed to clear session on server:", e);
        }
    }

    // Reset local state
    sessionId = null;
    conversationTurns = 0;

    // Clear chat UI
    chatContainer.innerHTML = '';

    // Add welcome message back
    const welcomeHtml = `
        <div class="flex gap-3 animate-fade">
            <div class="w-10 h-10 rounded-xl overflow-hidden flex-shrink-0">
                <img src="img/sentra-logo.png" alt="AI" class="w-full h-full object-cover" style="filter: sepia(100%) saturate(300%) hue-rotate(-10deg) brightness(0.7);">
            </div>
            <div class="bubble-bot p-4 rounded-2xl rounded-tl-none max-w-[85%]">
                <p class="chat-text font-semibold mb-1">New conversation started!</p>
                <p class="chat-text-secondary text-sm">I'm ready to help analyze media bias and detect AI hallucinations. What would you like to know?</p>
            </div>
        </div>
    `;
    chatContainer.innerHTML = welcomeHtml;

    // Reset sidebar - hide content, show empty state
    document.getElementById('sidebar-empty').classList.remove('hidden');
    document.getElementById('sidebar-content').classList.add('hidden');

    // Clear sources list
    const sourcesList = document.getElementById('sources-list');
    const sourceCount = document.getElementById('source-count');
    if (sourcesList) sourcesList.innerHTML = '<p class="text-slate-500 text-[10px] text-center py-4">No sources yet</p>';
    if (sourceCount) sourceCount.innerText = '(0)';

    // Clear framing comparison
    const framingDiv = document.getElementById('framing-comparison');
    if (framingDiv) framingDiv.innerHTML = '';

    // Clear hallucination lists
    const halluListA = document.getElementById('hallucination-list-a');
    const halluListB = document.getElementById('hallucination-list-b');
    if (halluListA) halluListA.innerHTML = '';
    if (halluListB) halluListB.innerHTML = '';

    updateConversationIndicator();
    console.log("[Session] New chat started - all data cleared");
}

function updateConversationIndicator() {
    // Update UI to show conversation turn count (optional visual feedback)
    const statusBadge = document.querySelector('.status-badge span:last-child');
    if (statusBadge && conversationTurns > 0) {
        statusBadge.textContent = `Turn ${conversationTurns}`;
    } else if (statusBadge) {
        statusBadge.textContent = 'Active';
    }
}

function appendMessage(text, sender) {
    const div = document.createElement('div');
    div.className = `flex gap-3 ${sender === 'user' ? 'flex-row-reverse' : ''} animate-fade`;

    const avatar = document.createElement('div');
    if (sender === 'user') {
        avatar.className = 'w-10 h-10 rounded-xl flex items-center justify-center text-sm font-bold flex-shrink-0';
        avatar.style.background = 'var(--btn-primary-bg)';
        avatar.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#F3E9DC" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`;
    } else {
        avatar.className = 'w-10 h-10 rounded-xl overflow-hidden flex-shrink-0';
        avatar.innerHTML = '<img src="img/sentra-logo.png" alt="AI" class="w-full h-full object-cover" style="filter: sepia(100%) saturate(300%) hue-rotate(-10deg) brightness(0.7);">';
    }

    const bubble = document.createElement('div');
    if (sender === 'user') {
        bubble.className = 'bubble-user p-4 rounded-2xl rounded-tr-none text-white text-sm max-w-[85%]';
        bubble.innerText = text;
    } else {
        bubble.className = 'bubble-bot p-4 rounded-2xl rounded-tl-none text-sm leading-relaxed max-w-[85%] response-content';
        bubble.innerHTML = text;
    }

    div.appendChild(avatar);
    div.appendChild(bubble);
    chatContainer.appendChild(div);

    // Smooth scroll to bottom
    chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
}

function appendLoading() {
    const id = 'loading-' + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = 'flex gap-3 animate-fade';
    div.innerHTML = `
        <div class="w-10 h-10 rounded-xl overflow-hidden flex-shrink-0">
            <img src="img/sentra-logo.png" alt="AI" class="w-full h-full object-cover" style="filter: sepia(100%) saturate(300%) hue-rotate(-10deg) brightness(0.7);">
        </div>
        <div class="bubble-bot p-4 rounded-2xl rounded-tl-none flex items-center gap-2">
            <div class="loading-dot w-2.5 h-2.5 rounded-full"></div>
            <div class="loading-dot w-2.5 h-2.5 rounded-full"></div>
            <div class="loading-dot w-2.5 h-2.5 rounded-full"></div>
            <span class="chat-text-secondary text-sm ml-2">Analyzing sources...</span>
        </div>
    `;
    chatContainer.appendChild(div);
    chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
    return id;
}

function updateSidebar(data) {
    const emptyState = document.getElementById('sidebar-empty');
    const content = document.getElementById('sidebar-content');

    emptyState.classList.add('hidden');
    content.classList.remove('hidden');

    const comparison = data.model_comparison || {};

    // Confidence
    if (comparison.confidence) {
        animateValue('confidence-a', comparison.confidence.model_a?.score_percent || '0%');
        animateValue('confidence-b', comparison.confidence.model_b?.score_percent || '0%');
        document.getElementById('confidence-a-method').innerText = comparison.confidence.model_a?.name || 'Random Forest';
        document.getElementById('confidence-b-method').innerText = comparison.confidence.model_b?.name || 'Heuristic';
    }

    // Hallucination
    if (comparison.hallucination) {
        const ratioA = comparison.hallucination.model_a?.supported_ratio || '0/0';
        const ratioB = comparison.hallucination.model_b?.supported_ratio || '0/0';

        document.getElementById('hallu-a-ratio').innerText = ratioA;
        document.getElementById('hallu-b-ratio').innerText = ratioB;

        updateLabel('hallu-a-label', ratioA);
        updateLabel('hallu-b-label', ratioB);
    }

    // Framing
    const framingDiv = document.getElementById('framing-comparison');
    framingDiv.innerHTML = '';

    if (comparison.framing) {
        const mediaList = Object.keys(comparison.framing.model_a?.keywords || {});

        mediaList.forEach(media => {
            const keywordsA = comparison.framing.model_a?.keywords[media] || [];
            const keywordsB = comparison.framing.model_b?.keywords[media] || [];

            const item = document.createElement('div');
            item.className = 'p-2 rounded-lg';
            item.style.cssText = 'background: var(--bg-secondary); border: 1px solid var(--border-color);';
            item.innerHTML = `
                <span class="text-[10px] font-bold uppercase" style="color: var(--color-chestnut);">${media.replace('_', ' ')}</span>
                <div class="grid grid-cols-2 gap-2 mt-1">
                    <div class="text-[10px] p-1.5 rounded" style="color: var(--text-secondary); background: var(--factual-bg); border: 1px solid var(--factual-border);">
                        <strong>A:</strong> ${keywordsA.join(', ') || 'N/A'}
                    </div>
                    <div class="text-[10px] p-1.5 rounded" style="color: var(--text-secondary); background: var(--glass-panel-bg); border: 1px solid var(--border-color);">
                        <strong>B:</strong> ${keywordsB.join(', ') || 'N/A'}
                    </div>
                </div>
            `;
            framingDiv.appendChild(item);
        });
    }

    // Hallucination Lists
    try {
        updateHallucinationList('hallucination-list-a', data.unsupported_claims || []);
        updateHallucinationList('hallucination-list-b', data.unsupported_claims_b || []);
    } catch (err) { console.error("Hallucination update failed", err); }

    // Sources
    try {
        updateSourcesList(data.sources || {});
    } catch (err) { console.error("Sources update failed", err); }
}

function animateValue(elementId, targetValue) {
    const el = document.getElementById(elementId);
    const target = parseInt(targetValue) || 0;
    const duration = 800;
    const start = parseInt(el.innerText) || 0;
    const startTime = performance.now();

    // Get corresponding progress circle
    const progressId = elementId === 'confidence-a' ? 'progress-a' : 'progress-b';
    const progressCircle = document.getElementById(progressId);
    const circumference = 220; // 2 * PI * 35

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Easing function for smoother animation
        const easeProgress = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(start + (target - start) * easeProgress);
        el.innerText = current + '%';

        // Update circular progress
        if (progressCircle) {
            const offset = circumference - (current / 100 * circumference);
            progressCircle.style.strokeDashoffset = offset;
        }

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

function updateHallucinationList(listId, claims) {
    const list = document.getElementById(listId);
    list.innerHTML = '';

    if (claims && claims.length > 0) {
        claims.slice(0, 3).forEach(claim => {
            const li = document.createElement('li');
            li.className = 'text-amber-300';
            li.innerText = `"${claim.sentence.substring(0, 30)}..."`;
            list.appendChild(li);
        });
    } else {
        list.innerHTML = '<li class="text-emerald-400">All claims verified ✓</li>';
    }
}

function updateLabel(elementId, ratioString) {
    const el = document.getElementById(elementId);
    const [unver, total] = ratioString.split('/').map(Number);

    if (unver > 0) {
        el.innerText = 'Weak Alignment';
        el.className = 'text-[10px] font-medium';
        el.style.color = 'var(--color-hallucination)';  // Red - Hallucination risk
    } else {
        el.innerText = 'Strong Alignment';
        el.className = 'text-[10px] font-medium';
        el.style.color = 'var(--color-factual)';  // Green - Factual confidence
    }
}

function updateSourcesList(sources) {
    const list = document.getElementById('sources-list');
    const count = document.getElementById('source-count');

    console.log("[DEBUG] Updating Sources List:", sources);

    if (!sources || typeof sources !== 'object') {
        console.warn("[WARN] Sources is invalid:", sources);
        list.innerHTML = '<p class="text-slate-500 text-[10px] text-center py-4">No sources data</p>';
        count.innerText = '(0)';
        return;
    }

    const keys = Object.keys(sources);
    // DEBUG: Dump raw structure to check data
    // list.innerHTML = `<pre class="text-[8px] text-slate-400 overflow-x-auto">${JSON.stringify(sources, null, 2)}</pre>`;

    if (keys.length === 0) {
        list.innerHTML = '<p class="text-slate-500 text-[10px] text-center py-4">No sources found (Empty Keys)</p>';
        count.innerText = '(0)';
        return;
    }

    list.innerHTML = '';
    let totalCount = 0;

    try {
        for (const [media, chunks] of Object.entries(sources)) {
            console.log(`[DEBUG] Processing media: ${media}, Chunks:`, chunks);

            if (Array.isArray(chunks) && chunks.length > 0) {
                chunks.forEach(chunk => {
                    try {
                        totalCount++;
                        const card = document.createElement('div');
                        const url = chunk.url || '#';
                        const hasUrl = url && url.startsWith('http');

                        card.className = `source-card glass-panel p-3 rounded-xl mb-2 ${hasUrl ? 'cursor-pointer hover:border-cyan-400/50' : ''}`;

                        const badgeColor = getBadgeColor(media);
                        const similarity = Math.round((chunk.similarity || 0) * 100);
                        const title = chunk.title || 'Untitled Article';
                        const text = chunk.text || 'No preview available';

                        card.innerHTML = `
                            <div class="flex items-center justify-between mb-2">
                                <span class="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-slate-700/50 ${badgeColor}">${formatMediaName(media)}</span>
                                <div class="flex items-center gap-2">
                                    <span class="text-[10px] text-cyan-400 font-medium">${similarity}%</span>
                                    ${hasUrl ? '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="text-cyan-400"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>' : ''}
                                </div>
                            </div>
                            <p class="text-xs text-white font-medium mb-1 line-clamp-1">${truncateText(title, 60)}</p>
                            <p class="text-[11px] text-slate-400 leading-relaxed line-clamp-2">${truncateText(text, 100)}</p>
                        `;

                        if (hasUrl) {
                            card.addEventListener('click', () => window.open(url, '_blank'));
                        }

                        list.appendChild(card);
                    } catch (innerErr) {
                        console.error("[ERROR] Rendering chunk:", innerErr, chunk);
                    }
                });
            } else {
                console.log(`[DEBUG] No chunks for ${media} or invalid array`);
            }
        }
        console.log("[DEBUG] Total sources rendered:", totalCount);
    } catch (e) {
        console.error("[ERROR] updating sources:", e);
    }

    count.innerText = `(${totalCount})`;
}

function getBadgeColor(media) {
    const lower = media.toLowerCase();
    if (lower.includes('antara')) return 'text-emerald-400';
    if (lower.includes('tempo')) return 'text-red-400';
    if (lower.includes('abc')) return 'text-cyan-400';
    return 'text-slate-400';
}

function formatMediaName(media) {
    const lower = media.toLowerCase();
    if (lower.includes('antara')) return 'ANTARA';
    if (lower.includes('tempo')) return 'Tempo';
    if (lower.includes('abc')) return 'ABC News';
    return media;
}

function truncateText(text, maxLen) {
    if (!text) return '';
    return text.length > maxLen ? text.substring(0, maxLen) + '...' : text;
}
