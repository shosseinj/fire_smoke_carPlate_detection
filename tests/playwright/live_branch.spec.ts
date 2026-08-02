import { execFileSync } from "node:child_process";
import { test, expect, type Locator, type Page } from "@playwright/test";

type VideoSnapshot = {
  readyState: number;
  paused: boolean;
  currentTime: number;
  videoWidth: number;
  videoHeight: number;
  clientWidth: number;
  clientHeight: number;
};

const snapshot = (video: Locator) =>
  video.evaluate((element): VideoSnapshot => ({
    readyState: element.readyState,
    paused: element.paused,
    currentTime: element.currentTime,
    videoWidth: element.videoWidth,
    videoHeight: element.videoHeight,
    clientWidth: element.clientWidth,
    clientHeight: element.clientHeight,
  }));

const evidenceByPage = new WeakMap<Page, { events: string[]; requests: string[]; failures: string[] }>();

test.afterEach(async ({ page }, testInfo) => {
  if (testInfo.status === testInfo.expectedStatus) return;
  const evidence = evidenceByPage.get(page);
  if (evidence) {
    await testInfo.attach("console-page-errors-api-whep", { body: evidence.events.concat(evidence.requests, evidence.failures).join("\n"), contentType: "text/plain" });
  }
  try {
    await testInfo.attach("failure-page", { body: await page.screenshot({ fullPage: true }), contentType: "image/png" });
    await testInfo.attach("failure-html", { body: await page.content(), contentType: "text/html" });
  } catch (_) { /* browser launch/navigation failures may have no page */ }
  try {
    const logs = execFileSync("docker", ["compose", "logs", "--no-color", "--tail", "200", "video-ai-router", "mediamtx"], { encoding: "utf8", timeout: 30_000 });
    await testInfo.attach("app-mediamtx-logs", { body: logs.replace(/rtsp:\/\/[^\s]+/gi, "rtsp://[redacted]"), contentType: "text/plain" });
  } catch (error) {
    await testInfo.attach("app-mediamtx-logs", { body: `docker compose logs unavailable: ${error}`, contentType: "text/plain" });
  }
});

test.describe("GPU live branch (runtime-dependent)", () => {
  test.skip(!process.env.LIVE_BRANCH_E2E, "Requires MediaMTX, DeepStream/NVMM, and a real source; playback is never mocked");

  test("connects, plays, and advances native media time", async ({ page }, testInfo) => {
    const events: string[] = [];
    const requests: string[] = [];
    const failures: string[] = [];
    evidenceByPage.set(page, { events, requests, failures });
    page.on("console", (message) => events.push(`console ${message.type()}: ${message.text()}`));
    page.on("pageerror", (error) => events.push(`pageerror: ${error.message}`));
    page.on("request", (request) => {
      if (request.method() === "POST" || request.method() === "DELETE" || request.url().includes("whep")) {
        requests.push(`${request.method()} ${request.url()}\n${request.postData() ?? ""}`);
      }
    });
    page.on("requestfailed", (request) => failures.push(`${request.method()} ${request.url()} :: ${request.failure()?.errorText}`));
    page.on("response", async (response) => {
      if (response.url().includes("/api/v1/live-branch/") || response.url().includes("/whep")) {
        let body = "";
        try { body = (await response.text()).slice(0, 4000); } catch (error) { body = `body unavailable: ${error}`; }
        requests.push(`RESPONSE ${response.status()} ${response.url()}\n${body}`);
      }
    });
    await page.addInitScript(() => {
      const NativePeer = window.RTCPeerConnection;
      const states: string[] = [];
      (window as unknown as { __livePeerStates: string[] }).__livePeerStates = states;
      const ObservedPeer = function (this: RTCPeerConnection, ...args: ConstructorParameters<typeof RTCPeerConnection>) {
        const peer = new NativePeer(...args);
        const record = () => states.push(`connection=${peer.connectionState} ice=${peer.iceConnectionState}`);
        peer.addEventListener("connectionstatechange", record);
        peer.addEventListener("iceconnectionstatechange", record);
        record();
        return peer;
      } as unknown as typeof RTCPeerConnection;
      ObservedPeer.prototype = NativePeer.prototype;
      window.RTCPeerConnection = ObservedPeer;
    });
    await page.goto(`${process.env.DASHBOARD_URL ?? "http://127.0.0.1:9999"}/dashboard`, { waitUntil: "domcontentloaded" });
    await page.getByRole("button", { name: "See live branch" }).click();
    await expect(page.locator("#liveBranchSection")).toBeVisible();
    const video = page.locator("#liveBranchGrid video").first();
    await expect(video).toBeVisible();
    await expect(video).toHaveJSProperty("clientWidth", 260);
    await expect(video).toHaveJSProperty("clientHeight", 260);
    await expect(video).toHaveJSProperty("autoplay", true);
    await expect(video).toHaveJSProperty("muted", true);
    await expect.poll(async () => (await snapshot(video)).readyState, { timeout: 45_000 }).toBeGreaterThanOrEqual(2);
    await expect.poll(async () => (await snapshot(video)).paused, { timeout: 45_000 }).toBe(false);
    const wallStart = await snapshot(video);
    await page.waitForTimeout(3_000);
    const wallEnd = await snapshot(video);
    expect(wallEnd.currentTime, "wall currentTime must advance; a URL alone is not proof of playback").toBeGreaterThan(wallStart.currentTime);
    expect(wallEnd.videoWidth).toBeGreaterThan(0);
    expect(wallEnd.videoHeight).toBeGreaterThan(0);
    expect(requests.some((request) => request.startsWith("POST") && request.includes("/acquire"))).toBeTruthy();
    expect(requests.some((request) => request.includes("/whep"))).toBeTruthy();
    const peerStates = await page.evaluate(() => (window as unknown as { __livePeerStates: string[] }).__livePeerStates);
    expect(peerStates.some((state) => state.includes("connection=connected"))).toBeTruthy();
    expect(peerStates.some((state) => state.includes("ice=connected") || state.includes("ice=completed"))).toBeTruthy();

    await video.click();
    await expect.poll(() => page.evaluate(() => Boolean(document.fullscreenElement)), { timeout: 10_000 }).toBe(true);
    const fullscreenStart = await snapshot(video);
    expect(fullscreenStart.videoWidth).toBeGreaterThan(0);
    expect(fullscreenStart.videoHeight).toBeGreaterThan(0);
    await page.waitForTimeout(3_000);
    const fullscreenEnd = await snapshot(video);
    expect(fullscreenEnd.currentTime, "native fullscreen currentTime must advance").toBeGreaterThan(fullscreenStart.currentTime);
    const nativeWidth = Number(await video.locator("..").getAttribute("data-native-width"));
    const nativeHeight = Number(await video.locator("..").getAttribute("data-native-height"));
    if (nativeWidth > 0 && nativeHeight > 0) {
      expect(fullscreenEnd.videoWidth).toBe(nativeWidth);
      expect(fullscreenEnd.videoHeight).toBe(nativeHeight);
    }
    expect(requests.some((request) => request.startsWith("POST") && request.includes("/fullscreen/acquire"))).toBeTruthy();

    await page.keyboard.press("Escape");
    await expect.poll(() => page.evaluate(() => Boolean(document.fullscreenElement))).toBe(false);
    await page.getByRole("button", { name: "بازگشت به پیش‌نمایش" }).click();
    await expect(page.locator("#liveBranchSection")).toBeHidden();
    await expect.poll(() => page.locator("#liveBranchGrid video").count()).toBe(0);
    expect(requests.filter((request) => request.startsWith("DELETE") && request.includes("/release")).length).toBeGreaterThanOrEqual(2);
    expect(failures, failures.join("\n")).toEqual([]);
    await testInfo.attach("runtime-observations", { body: [...events, ...requests, ...failures].join("\n"), contentType: "text/plain" });
  });
});
