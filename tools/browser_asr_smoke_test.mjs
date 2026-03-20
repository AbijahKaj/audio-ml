import puppeteer from 'puppeteer-core';

const url = process.argv[2] ?? 'http://127.0.0.1:4173/?autotest=1#speech-recognizer';

const browser = await puppeteer.launch({
  executablePath: '/usr/local/bin/google-chrome',
  headless: true,
  args: ['--no-sandbox', '--disable-gpu'],
});

try {
  const page = await browser.newPage();
  page.on('console', (message) => {
    console.log(`[browser:${message.type()}] ${message.text()}`);
  });
  page.on('pageerror', (error) => {
    console.log(`[pageerror] ${error.message}`);
  });

  await page.goto(url, { waitUntil: 'networkidle2', timeout: 120000 });

  await page.waitForFunction(
    () => document.body.dataset.asrAutotest === 'pass' || document.body.dataset.asrAutotest === 'fail',
    { timeout: 300000 },
  );

  const result = await page.evaluate(() => ({
    autotest: document.body.dataset.asrAutotest ?? 'missing',
    transcript: document.body.dataset.asrTranscript ?? '',
    status: document.getElementById('speech-recognizer-status')?.textContent ?? '',
    finalText: document.getElementById('speech-recognizer-final')?.textContent ?? '',
    log: document.getElementById('speech-recognizer-log')?.textContent ?? '',
  }));

  console.log(JSON.stringify(result, null, 2));

  if (result.autotest !== 'pass') {
    process.exitCode = 1;
  }
} finally {
  await browser.close();
}
