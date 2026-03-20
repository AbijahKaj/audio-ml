import type { DecoderType } from '../model/ModelConfig';

interface BridgeReadyResponse {
  event: 'ready';
  sample_rate: number;
  decoder_type: DecoderType;
  model_name: string;
}

interface BridgeOkResponse {
  id: number;
  ok: true;
  text?: string;
}

interface BridgeErrorResponse {
  id: number;
  ok: false;
  error: string;
  traceback?: string;
}

type BridgeResponse = BridgeOkResponse | BridgeErrorResponse | BridgeReadyResponse;

interface PendingRequest {
  resolve: (value: BridgeOkResponse) => void;
  reject: (reason: Error) => void;
}

export interface NemoBridgeConfig {
  pythonPath?: string;
  scriptPath?: string;
}

export interface NemoBridgeReadyInfo {
  sampleRate: number;
  decoderType: DecoderType;
  modelName: string;
}

/**
 * Thin JSON-line bridge to a persistent Python NeMo process.
 * Node-only helper used for real-model compatibility.
 */
export class NemoPythonBridge {
  private processHandle: any | null = null;
  private pending = new Map<number, PendingRequest>();
  private nextRequestId = 1;
  private readyInfo: NemoBridgeReadyInfo | null = null;
  private stopping = false;

  constructor(private config: NemoBridgeConfig = {}) {}

  async start(modelName: string): Promise<NemoBridgeReadyInfo> {
    if (this.processHandle) {
      if (this.readyInfo) return this.readyInfo;
      throw new Error('NeMo bridge process already started but not ready.');
    }

    const spawn = await this.loadSpawn();
    const scriptPath = this.config.scriptPath ?? '/workspace/tools/nemo_bridge_server.py';
    const pythonPath = this.config.pythonPath ?? 'python3';
    const child = spawn(pythonPath, [scriptPath, '--model', modelName], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: process.env,
    });
    this.processHandle = child;

    this.attachStdErr(child);
    this.attachExitHandler(child);

    const ready = await this.waitForReady(child, 180000);
    this.readyInfo = ready;
    return ready;
  }

  async transcribePcm(pcm: Float32Array, sampleRate: number): Promise<string> {
    const response = await this.request({
      cmd: 'transcribe',
      sample_rate: sampleRate,
      pcm: Array.from(pcm),
    });
    return response.text ?? '';
  }

  async stop(): Promise<void> {
    if (!this.processHandle) return;
    this.stopping = true;
    try {
      await this.request({ cmd: 'shutdown' });
    } catch {
      // Best-effort shutdown.
    }
    if (this.processHandle && !this.processHandle.killed) {
      this.processHandle.kill();
    }
    this.processHandle = null;
    this.readyInfo = null;
    this.rejectAllPending(new Error('NeMo bridge stopped.'));
    this.stopping = false;
  }

  private async request(payload: Record<string, unknown>): Promise<BridgeOkResponse> {
    if (!this.processHandle) {
      throw new Error('NeMo bridge is not started.');
    }

    const requestId = this.nextRequestId++;
    const request = { id: requestId, ...payload };

    const promise = new Promise<BridgeOkResponse>((resolve, reject) => {
      this.pending.set(requestId, { resolve, reject });
    });

    this.processHandle.stdin.write(`${JSON.stringify(request)}\n`);
    return promise;
  }

  private async waitForReady(child: any, timeoutMs: number): Promise<NemoBridgeReadyInfo> {
    const readline = await import('node:readline');
    const rl = readline.createInterface({ input: child.stdout });

    return new Promise<NemoBridgeReadyInfo>((resolve, reject) => {
      const timeout = setTimeout(() => {
        rl.close();
        reject(new Error(`Timed out waiting for NeMo bridge ready after ${timeoutMs}ms.`));
      }, timeoutMs);

      rl.on('line', (line: string) => {
        const parsed = this.parseBridgeLine(line);
        if (!parsed) return;

        if ('event' in parsed && parsed.event === 'ready') {
          clearTimeout(timeout);
          const info: NemoBridgeReadyInfo = {
            sampleRate: parsed.sample_rate,
            decoderType: parsed.decoder_type,
            modelName: parsed.model_name,
          };
          rl.on('line', (nextLine: string) => this.handleResponseLine(nextLine));
          resolve(info);
        } else {
          this.resolveResponse(parsed);
        }
      });

      rl.on('error', (error: Error) => {
        clearTimeout(timeout);
        reject(error);
      });
    });
  }

  private handleResponseLine(line: string): void {
    const parsed = this.parseBridgeLine(line);
    if (!parsed) return;
    this.resolveResponse(parsed);
  }

  private resolveResponse(message: BridgeResponse): void {
    if (!('id' in message)) return;
    const request = this.pending.get(message.id);
    if (!request) return;
    this.pending.delete(message.id);

    if ('ok' in message && message.ok) {
      request.resolve(message);
      return;
    }

    if ('ok' in message && !message.ok) {
      const trace = message.traceback ? `\n${message.traceback}` : '';
      request.reject(new Error(`NeMo bridge error: ${message.error}${trace}`));
    }
  }

  private parseBridgeLine(line: string): BridgeResponse | null {
    const trimmed = line.trim();
    if (!trimmed.startsWith('{')) {
      return null;
    }
    try {
      return JSON.parse(trimmed) as BridgeResponse;
    } catch {
      return null;
    }
  }

  private attachStdErr(child: any): void {
    child.stderr.on('data', (chunk: Uint8Array) => {
      const message = new TextDecoder().decode(chunk).trim();
      if (message.length > 0) {
        // Surface bridge logs as console warnings for diagnostics.
        console.warn(message);
      }
    });
  }

  private attachExitHandler(child: any): void {
    child.on('exit', (code: number | null, signal: string | null) => {
      if (this.stopping) return;
      const reason = `NeMo bridge exited (code=${String(code)}, signal=${String(signal)})`;
      this.rejectAllPending(new Error(reason));
      this.processHandle = null;
      this.readyInfo = null;
    });
  }

  private rejectAllPending(error: Error): void {
    for (const pending of this.pending.values()) {
      pending.reject(error);
    }
    this.pending.clear();
  }

  private async loadSpawn(): Promise<any> {
    const childProcess = await import('node:child_process');
    return childProcess.spawn;
  }
}
