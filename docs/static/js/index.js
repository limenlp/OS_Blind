document.addEventListener('DOMContentLoaded', async () => {
  initNavbarBurger();
  await initShowcase();
});

function initNavbarBurger() {
  const burger = document.querySelector('.navbar-burger');
  const menu = document.querySelector('.navbar-menu');

  if (!burger || !menu) {
    return;
  }

  burger.addEventListener('click', () => {
    burger.classList.toggle('is-active');
    menu.classList.toggle('is-active');
  });
}

async function loadShowcaseCases() {
  const response = await fetch('./static/data/showcase-cases.json', { cache: 'no-store' });

  if (!response.ok) {
    throw new Error(`Failed to load showcase data: ${response.status}`);
  }

  return response.json();
}

async function initShowcase() {
  const tabsEl = document.getElementById('showcase-tabs');
  const instructionEl = document.getElementById('traj-instruction');
  const riskEl = document.getElementById('traj-risk');
  const stepListEl = document.getElementById('showcase-step-list');
  const imageEl = document.getElementById('traj-image');
  const prevBtn = document.getElementById('prev-step');
  const nextBtn = document.getElementById('next-step');
  const playBtn = document.getElementById('play-steps');
  const replayBtn = document.getElementById('replay-step');

  if (!tabsEl || !instructionEl || !riskEl || !stepListEl || !imageEl || !prevBtn || !nextBtn || !playBtn || !replayBtn) {
    return;
  }

  let showcaseCases;
  try {
    showcaseCases = await loadShowcaseCases();
  } catch (error) {
    console.error(error);
    return;
  }

  if (!Array.isArray(showcaseCases) || showcaseCases.length === 0) {
    return;
  }

  showcaseCases = showcaseCases.slice(3);

  const showcaseOrder = [
    'fake_news',
    'security_check',
    'term_5',
    'api_config_theft',
    'aggressive_10',
  ];
  showcaseCases = showcaseOrder
    .map((id) => showcaseCases.find((item) => item.id === id))
    .filter(Boolean);

  if (showcaseCases.length === 0) {
    return;
  }

  let currentCaseIndex = 0;
  let currentFrameIndex = 0;
  let playTimer = null;

  const escapeHtml = (value) => String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');

  const stopPlayback = () => {
    if (playTimer) {
      window.clearInterval(playTimer);
      playTimer = null;
    }

    playBtn.setAttribute('aria-pressed', 'false');
    playBtn.innerHTML = '<i class="fas fa-play" aria-hidden="true"></i>';
  };

  const getCurrentCase = () => showcaseCases[currentCaseIndex];

  const getActiveStepIndex = () => {
    const currentCase = getCurrentCase();
    return Math.min(currentFrameIndex, currentCase.steps.length - 1);
  };

  const getCurrentImage = () => {
    const currentCase = getCurrentCase();
    if (currentFrameIndex === 0) {
      return currentCase.initialImage;
    }
    return currentCase.steps[currentFrameIndex - 1].image;
  };

  const renderTabs = () => {
    tabsEl.innerHTML = showcaseCases.map((item, index) => `
      <li class="trajectory-tab${index === currentCaseIndex ? ' is-active' : ''}"
          data-case-index="${index}"
          role="tab"
          aria-selected="${index === currentCaseIndex}">
        <a>${escapeHtml(item.label)}</a>
      </li>
    `).join('');
  };

  const renderSteps = () => {
    const currentCase = getCurrentCase();
    const activeStepIndex = getActiveStepIndex();

    stepListEl.innerHTML = currentCase.steps.map((step, index) => `
      <li class="step-list-item${index === activeStepIndex ? ' active' : ''}" data-step-index="${index}">
        <div class="step-header">
          <div class="step-left">
            <span class="step-number">Step ${index + 1}</span>
            <span class="step-title">${escapeHtml(step.title)}</span>
          </div>
        </div>
        <div class="step-action-details">
          <code>
            <span class="action-type">${escapeHtml(step.actionType)}</span>
            <span class="action-copy">${escapeHtml(step.actionText)}</span>
          </code>
          <details class="step-thought">
            <summary>Thought</summary>
            <pre>${escapeHtml(step.thought)}</pre>
          </details>
        </div>
      </li>
    `).join('');

    const activeStep = stepListEl.querySelector('.step-list-item.active');
    if (activeStep) {
      activeStep.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  };

  const renderCurrentFrame = () => {
    const currentCase = getCurrentCase();
    const activeStepIndex = getActiveStepIndex();

    instructionEl.textContent = currentCase.instruction;
    riskEl.textContent = currentCase.riskSummary;
    imageEl.src = getCurrentImage();
    imageEl.alt = currentFrameIndex === 0
      ? `${currentCase.label} initial screen`
      : `${currentCase.label} post-step ${currentFrameIndex} screen`;

    imageEl.classList.remove('is-animating');
    window.requestAnimationFrame(() => imageEl.classList.add('is-animating'));

    renderTabs();
    renderSteps();

    prevBtn.disabled = currentFrameIndex === 0;
    nextBtn.disabled = currentFrameIndex === currentCase.steps.length;
    imageEl.dataset.activeStep = String(activeStepIndex + 1);
  };

  const resetCase = () => {
    stopPlayback();
    currentFrameIndex = 0;
    renderCurrentFrame();
  };

  const setCase = (nextCaseIndex) => {
    stopPlayback();
    currentCaseIndex = nextCaseIndex;
    currentFrameIndex = 0;
    renderCurrentFrame();
  };

  const setStep = (nextStepIndex) => {
    stopPlayback();
    currentFrameIndex = nextStepIndex;
    renderCurrentFrame();
  };

  const goNext = () => {
    const currentCase = getCurrentCase();

    if (currentFrameIndex >= currentCase.steps.length) {
      stopPlayback();
      return false;
    }

    currentFrameIndex += 1;
    renderCurrentFrame();
    return true;
  };

  const goPrev = () => {
    if (currentFrameIndex === 0) {
      return false;
    }

    currentFrameIndex -= 1;
    renderCurrentFrame();
    return true;
  };

  tabsEl.addEventListener('click', (event) => {
    const tab = event.target.closest('.trajectory-tab');
    if (!tab) {
      return;
    }

    setCase(Number(tab.dataset.caseIndex));
  });

  stepListEl.addEventListener('click', (event) => {
    if (event.target.closest('.step-thought')) {
      return;
    }

    const stepItem = event.target.closest('.step-list-item');
    if (!stepItem) {
      return;
    }

    setStep(Number(stepItem.dataset.stepIndex));
  });

  prevBtn.addEventListener('click', () => {
    stopPlayback();
    goPrev();
  });

  nextBtn.addEventListener('click', () => {
    stopPlayback();
    goNext();
  });

  replayBtn.addEventListener('click', () => {
    resetCase();
  });

  playBtn.addEventListener('click', () => {
    if (playTimer) {
      stopPlayback();
      return;
    }

    const currentCase = getCurrentCase();
    if (currentFrameIndex === currentCase.steps.length) {
      currentFrameIndex = 0;
      renderCurrentFrame();
    }

    playBtn.setAttribute('aria-pressed', 'true');
    playBtn.innerHTML = '<i class="fas fa-pause" aria-hidden="true"></i>';

    playTimer = window.setInterval(() => {
      const advanced = goNext();
      if (!advanced) {
        stopPlayback();
      }
    }, 2400);
  });

  renderCurrentFrame();
}
