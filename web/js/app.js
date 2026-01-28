
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const chatContainer = document.getElementById('chat-container');
const analysisDashboard = document.getElementById('analysis-dashboard');
const analysisEmpty = document.getElementById('analysis-empty');

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = userInput.value.trim();
    if (!message) return;

    // Add User Message
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
            body: JSON.stringify({ message, mode })
        });

        const data = await response.json();

        // Remove loading
        document.getElementById(loadingId).remove();

        // Add Assistant Message
        appendMessage(marked.parse(data.answer), 'bot');

        // Update Sidebar with Comparison
        updateDashboard(data);

    } catch (error) {
        console.error(error);
        document.getElementById(loadingId).remove();
        appendMessage("Sorry, there was a connection error.", 'bot');
    }
});

function appendMessage(text, sender) {
    const div = document.createElement('div');
    div.className = `flex gap-4 ${sender === 'user' ? 'flex-row-reverse' : ''}`;

    const avatar = document.createElement('div');
    avatar.className = `w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0 ${sender === 'user' ? 'bg-slate-600 text-white' : 'bg-gradient-to-br from-indigo-500 to-purple-600 text-white'}`;
    avatar.innerText = sender === 'user' ? 'U' : 'AI';

    const bubble = document.createElement('div');
    bubble.className = `chat-bubble p-4 rounded-2xl shadow-lg text-sm leading-relaxed ${sender === 'user' ? 'bg-indigo-600 text-white rounded-tr-none' : 'bg-slate-800 border border-slate-700 text-slate-300 rounded-tl-none'}`;

    if (sender === 'bot') {
        bubble.innerHTML = text;
    } else {
        bubble.innerText = text;
    }

    div.appendChild(avatar);
    div.appendChild(bubble);
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function appendLoading() {
    const id = 'loading-' + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = 'flex gap-4';
    div.innerHTML = `
        <div class="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-sm font-bold flex-shrink-0">AI</div>
        <div class="bg-slate-800 border border-slate-700 p-4 rounded-2xl rounded-tl-none shadow-lg flex items-center gap-2">
            <div class="w-2 h-2 bg-indigo-400 rounded-full animate-bounce"></div>
            <div class="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
            <div class="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style="animation-delay: 0.4s"></div>
        </div>
    `;
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return id;
}

function updateDashboard(data) {
    analysisEmpty.classList.add('hidden');
    analysisDashboard.classList.remove('hidden');

    const comparison = data.model_comparison || {};

    // Confidence Comparison
    if (comparison.confidence) {
        document.getElementById('confidence-a').innerText = comparison.confidence.model_a?.score_percent || '0%';
        document.getElementById('confidence-b').innerText = comparison.confidence.model_b?.score_percent || '0%';
        document.getElementById('confidence-a-method').innerText = comparison.confidence.model_a?.name || 'ML Model';
        document.getElementById('confidence-b-method').innerText = comparison.confidence.model_b?.name || 'Baseline';
    }

    // Hallucination Comparison
    if (comparison.hallucination) {
        const ratioA = comparison.hallucination.model_a?.supported_ratio || '0/0';
        const ratioB = comparison.hallucination.model_b?.supported_ratio || '0/0';

        document.getElementById('hallu-a-ratio').innerText = ratioA;
        document.getElementById('hallu-b-ratio').innerText = ratioB;

        // Update Labels Logic
        updateLabel('hallu-a-label', ratioA);
        updateLabel('hallu-b-label', ratioB);
    }

    // Framing Keywords Comparison
    const framingDiv = document.getElementById('framing-comparison');
    framingDiv.innerHTML = '';

    if (comparison.framing) {
        const mediaList = Object.keys(comparison.framing.model_a?.keywords || {});

        mediaList.forEach(media => {
            const keywordsA = comparison.framing.model_a?.keywords[media] || [];
            const keywordsB = comparison.framing.model_b?.keywords[media] || [];

            const item = document.createElement('div');
            item.className = 'bg-slate-800 p-2 rounded-lg';
            item.innerHTML = `
                <span class="text-[10px] font-bold uppercase text-slate-400">${media.replace('_', ' ')}</span>
                <div class="grid grid-cols-2 gap-2 mt-1">
                    <div class="text-[10px] text-indigo-300 bg-indigo-500/10 p-1.5 rounded">
                        <strong>A:</strong> ${keywordsA.join(', ') || 'N/A'}
                    </div>
                    <div class="text-[10px] text-slate-400 bg-slate-700 p-1.5 rounded">
                        <strong>B:</strong> ${keywordsB.join(', ') || 'N/A'}
                    </div>
                </div>
            `;
            framingDiv.appendChild(item);
        });
    }

    // Hallucination List - Model A
    const halluListA = document.getElementById('hallucination-list-a');
    halluListA.innerHTML = '';

    if (data.unsupported_claims && data.unsupported_claims.length > 0) {
        data.unsupported_claims.slice(0, 4).forEach(claim => {
            const li = document.createElement('li');
            li.innerText = `"${claim.sentence.substring(0, 40)}..."`;
            halluListA.appendChild(li);
        });
    } else {
        halluListA.innerHTML = '<li class="text-emerald-400">All claims verified ✓</li>';
    }

    // Hallucination List - Model B
    const halluListB = document.getElementById('hallucination-list-b');
    halluListB.innerHTML = '';

    if (data.unsupported_claims_b && data.unsupported_claims_b.length > 0) {
        data.unsupported_claims_b.slice(0, 4).forEach(claim => {
            const li = document.createElement('li');
            li.innerText = `"${claim.sentence.substring(0, 40)}..."`;
            halluListB.appendChild(li);
        });
    } else {
        halluListB.innerHTML = '<li class="text-emerald-400">All claims verified ✓</li>';
    }
}

function updateLabel(elementId, ratioString) {
    const el = document.getElementById(elementId);
    const [unver, total] = ratioString.split('/').map(Number);

    // Now format is Unsupported / Total
    // 0 is good, > 0 is bad
    
    if (unver > 0) {
        el.innerText = 'Weak Source Alignment';
        el.classList.remove('text-emerald-400', 'text-slate-400', 'text-slate-500');
        el.classList.add('text-amber-400');
    } else {
        el.innerText = 'Strong Source Alignment';
        el.classList.remove('text-amber-400', 'text-slate-400', 'text-slate-500');
        el.classList.add('text-emerald-400');
    }
}
