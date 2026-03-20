import { Router, type DemoPage } from './router';
import { createAnalyzersDemo } from './pages/AnalyzersDemo';
import { createVADDemo } from './pages/VADDemo';
import { createAudioDenoiserDemo } from './pages/AudioDenoiserDemo';
import { createBeepDetectorDemo } from './pages/BeepDetectorDemo';
import { createSpeechRecognizerDemo } from './pages/SpeechRecognizerDemo';

let currentCleanup: (() => void) | null = null;
const router = new Router();

const pages: { page: DemoPage; label: string }[] = [
  { page: 'analyzers', label: 'All Analyzers' },
  { page: 'vad', label: 'VAD' },
  { page: 'denoiser', label: 'Denoiser' },
  { page: 'beep-detector', label: 'Voicemail Beep Detector' },
  { page: 'asr', label: 'ASR (FastConformer)' },
];

// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', () => {
  const appDiv = document.getElementById("app");
  if (!appDiv) {
    console.error("App div not found");
    return;
  }

  // Create navigation
  createNavigation(appDiv);

  // Create content container
  const contentContainer = document.createElement('div');
  contentContainer.id = 'demo-content';
  appDiv.appendChild(contentContainer);

  // Load initial page
  loadPage(router.getCurrentPage(), contentContainer);

  // Handle navigation (including browser back/forward)
  router.onNavigate((page) => {
    if (currentCleanup) {
      currentCleanup();
      currentCleanup = null;
    }
    contentContainer.innerHTML = '';
    loadPage(page, contentContainer);

    // Sync active nav button state (for browser back/forward)
    document.querySelectorAll('.demo-nav-button').forEach(btn => {
      btn.classList.toggle('active', (btn as HTMLElement).dataset.page === page);
    });
  });
});

function createNavigation(container: HTMLElement): void {
  const nav = document.createElement('nav');
  nav.className = 'demo-nav';

  const navTitle = document.createElement('h1');
  navTitle.textContent = 'Audio ML Demo';
  navTitle.className = 'demo-nav-title';
  nav.appendChild(navTitle);

  const navButtons = document.createElement('div');
  navButtons.className = 'demo-nav-buttons';

  pages.forEach(({ page, label }) => {
    const button = document.createElement('button');
    button.textContent = label;
    button.className = 'demo-nav-button';
    
    // Highlight current page
    if (router.getCurrentPage() === page) {
      button.classList.add('active');
    }

    button.dataset.page = page;
    button.addEventListener('click', () => {
      router.navigate(page);
      navButtons.querySelectorAll('button').forEach(btn => {
        btn.classList.remove('active');
      });
      button.classList.add('active');
    });

    navButtons.appendChild(button);
  });

  nav.appendChild(navButtons);
  container.appendChild(nav);
}

function loadPage(page: DemoPage, container: HTMLElement): void {
  switch (page) {
    case 'analyzers':
      currentCleanup = createAnalyzersDemo(container);
      break;
    case 'vad':
      currentCleanup = createVADDemo(container);
      break;
    case 'denoiser':
      currentCleanup = createAudioDenoiserDemo(container);
      break;
    case 'beep-detector':
      currentCleanup = createBeepDetectorDemo(container);
      break;
    case 'asr':
      currentCleanup = createSpeechRecognizerDemo(container);
      break;
    default:
      currentCleanup = createAnalyzersDemo(container);
  }
}
