/**
 * Base visualization classes
 */

export interface VisualizationOptions {
  width?: number;
  height?: number;
  color?: string;
  backgroundColor?: string;
  lineWidth?: number;
}

/**
 * Base visualization utility with common helpers
 */
export class BaseVisualizer {
  protected ctx: CanvasRenderingContext2D;
  protected width: number;
  protected height: number;
  protected color: string;
  protected backgroundColor: string;

  constructor(
    canvas: HTMLCanvasElement,
    options: VisualizationOptions = {}
  ) {
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get 2D context from canvas');
    this.ctx = ctx;
    this.width = options.width || canvas.width;
    this.height = options.height || canvas.height;
    this.color = options.color || '#00ff00';
    this.backgroundColor = options.backgroundColor || '#000000';
    
    canvas.width = this.width;
    canvas.height = this.height;
  }

  protected clear() {
    this.ctx.fillStyle = this.backgroundColor;
    this.ctx.fillRect(0, 0, this.width, this.height);
  }

  protected normalizeValue(value: number, min: number, max: number): number {
    if (max === min) return 0;
    return Math.max(0, Math.min(1, (value - min) / (max - min)));
  }
}

/**
 * Visualize array-based features (bars/line chart)
 */
export class ArrayVisualizer extends BaseVisualizer {
  constructor(
    canvas: HTMLCanvasElement,
    options: VisualizationOptions & { maxHistoryLength?: number } = {}
  ) {
    super(canvas, options);
  }

  /**
   * Draw array as vertical bars
   */
  drawBars(data: number[] | Float32Array, normalize: boolean = true) {
    this.clear();
    const len = data.length;
    if (len === 0) return;

    let min = 0, max = 1;
    if (normalize) {
      const values = Array.from(data);
      min = Math.min(...values);
      max = Math.max(...values);
    }

    const barWidth = this.width / len;
    this.ctx.fillStyle = this.color;

    for (let i = 0; i < len; i++) {
      const value = normalize ? this.normalizeValue(data[i], min, max) : data[i];
      const barHeight = value * this.height;
      this.ctx.fillRect(i * barWidth, this.height - barHeight, barWidth, barHeight);
    }
  }

  /**
   * Draw array as line chart
   */
  drawLine(data: number[] | Float32Array, normalize: boolean = true) {
    this.clear();
    const len = data.length;
    if (len === 0) return;

    let min = 0, max = 1;
    if (normalize) {
      const values = Array.from(data);
      min = Math.min(...values);
      max = Math.max(...values);
    }

    const stepX = this.width / (len - 1);
    this.ctx.strokeStyle = this.color;
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();

    for (let i = 0; i < len; i++) {
      const value = normalize ? this.normalizeValue(data[i], min, max) : data[i];
      const x = i * stepX;
      const y = this.height - (value * this.height);
      if (i === 0) {
        this.ctx.moveTo(x, y);
      } else {
        this.ctx.lineTo(x, y);
      }
    }
    this.ctx.stroke();
  }

  /**
   * Draw scrolling spectrogram (time-frequency)
   */
  drawSpectrogram(data: number[] | Float32Array) {
    const len = data.length;
    if (len === 0) return;

    // Shift canvas left by copying pixels
    // Use drawImage for better performance and to avoid CORS issues
    try {
      // Create a temporary canvas to hold the current image
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = this.width;
      tempCanvas.height = this.height;
      const tempCtx = tempCanvas.getContext('2d');
      if (tempCtx) {
        // Copy current canvas to temp
        tempCtx.drawImage(this.ctx.canvas, 0, 0);
        // Clear main canvas
        this.clear();
        // Draw shifted version (shift left by 1 pixel)
        this.ctx.drawImage(tempCanvas, -1, 0);
      }
    } catch (e) {
      // If drawImage fails, try imageData method
      try {
        const imageData = this.ctx.getImageData(1, 0, this.width - 1, this.height);
        this.ctx.putImageData(imageData, 0, 0);
        // Clear the rightmost column
        this.ctx.fillStyle = this.backgroundColor;
        this.ctx.fillRect(this.width - 1, 0, 1, this.height);
      } catch (e2) {
        // If both fail, just clear and draw (first frame)
        // Don't clear on subsequent calls
      }
    }

    // Draw new column on the right
    const bandHeight = this.height / len;
    const values = Array.from(data);
    const min = Math.min(...values);
    const max = Math.max(...values);

    for (let i = 0; i < len; i++) {
      const normalized = this.normalizeValue(values[i], min, max);
      const color = Math.floor(normalized * 255);
      this.ctx.fillStyle = `rgb(${color},${color},${color})`;
      this.ctx.fillRect(
        this.width - 1,
        this.height - (i + 1) * bandHeight,
        1,
        bandHeight
      );
    }
  }
}

/**
 * Visualize scalar values (time series)
 */
export class ScalarVisualizer extends BaseVisualizer {
  private history: number[] = [];
  private maxHistoryLength: number;

  constructor(
    canvas: HTMLCanvasElement,
    options: VisualizationOptions & { maxHistoryLength?: number } = {}
  ) {
    super(canvas, options);
    this.maxHistoryLength = options.maxHistoryLength || 200;
  }

  /**
   * Draw scalar value as time series line
   */
  drawTimeSeries(value: number, min?: number, max?: number) {
    this.history.push(value);
    if (this.history.length > this.maxHistoryLength) {
      this.history.shift();
    }

    this.clear();
    const len = this.history.length;
    if (len === 0) return;

    const actualMin = min !== undefined ? min : Math.min(...this.history);
    const actualMax = max !== undefined ? max : Math.max(...this.history);

    const stepX = this.width / (len - 1);
    this.ctx.strokeStyle = this.color;
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();

    for (let i = 0; i < len; i++) {
      const normalized = this.normalizeValue(this.history[i], actualMin, actualMax);
      const x = i * stepX;
      const y = this.height - (normalized * this.height);
      if (i === 0) {
        this.ctx.moveTo(x, y);
      } else {
        this.ctx.lineTo(x, y);
      }
    }
    this.ctx.stroke();

    // Draw current value as text
    this.ctx.fillStyle = this.color;
    this.ctx.font = '12px monospace';
    this.ctx.fillText(value.toFixed(2), 10, 20);
  }

  /**
   * Draw scalar value as gauge/meter
   */
  drawGauge(value: number, min: number, max: number, label?: string) {
    this.clear();
    const normalized = this.normalizeValue(value, min, max);
    const angle = normalized * Math.PI;
    const centerX = this.width / 2;
    const centerY = this.height;
    const radius = Math.min(this.width, this.height) * 0.8;

    // Draw arc
    this.ctx.strokeStyle = this.backgroundColor;
    this.ctx.lineWidth = 20;
    this.ctx.beginPath();
    this.ctx.arc(centerX, centerY, radius, Math.PI, 0, false);
    this.ctx.stroke();

    // Draw value arc
    this.ctx.strokeStyle = this.color;
    this.ctx.lineWidth = 20;
    this.ctx.beginPath();
    this.ctx.arc(centerX, centerY, radius, Math.PI, Math.PI - angle, true);
    this.ctx.stroke();

    // Draw value text
    if (label) {
      this.ctx.fillStyle = this.color;
      this.ctx.font = '14px monospace';
      this.ctx.textAlign = 'center';
      this.ctx.fillText(label, centerX, centerY - radius - 30);
      this.ctx.fillText(value.toFixed(2), centerX, centerY - radius - 10);
    }
  }
}
