/**
 * Simple router for demo pages
 */

export type DemoPage = 'analyzers' | 'vad' | 'denoiser' | 'beep-detector' | 'asr';

export class Router {
  private currentPage: DemoPage = 'analyzers';
  private listeners: Set<(page: DemoPage) => void> = new Set();

  constructor() {
    // Initial page from hash
    const initialPage = this.getPageFromHash();
    if (initialPage) {
      this.currentPage = initialPage;
    }

    // Handle browser back/forward
    window.addEventListener('popstate', () => {
      const page = this.getPageFromHash();
      if (page && page !== this.currentPage) {
        this.currentPage = page;
        this.notifyListeners();
      }
    });
  }

  private getPageFromHash(): DemoPage | null {
    const hash = window.location.hash.slice(1);
    if (hash === 'vad' || hash === 'denoiser' || hash === 'beep-detector' || hash === 'analyzers' || hash === 'asr') {
      return hash as DemoPage;
    }
    return null;
  }

  navigate(page: DemoPage): void {
    if (page !== this.currentPage) {
      this.currentPage = page;
      window.history.pushState({ page }, '', `#${page}`);
      this.notifyListeners();
    }
  }

  getCurrentPage(): DemoPage {
    return this.currentPage;
  }

  onNavigate(listener: (page: DemoPage) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notifyListeners(): void {
    this.listeners.forEach(listener => listener(this.currentPage));
  }
}
