"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";

type Quality = "AUTO" | "ULTRA" | "HIGH" | "BALANCED" | "ECO";

type Telemetry = {
  fps: number;
  drops: number;
  profile: string;
  gpu: string;
  adaptive: boolean;
};

const qualityDrops: Record<Exclude<Quality, "AUTO">, number> = {
  ULTRA: 70000,
  HIGH: 50000,
  BALANCED: 32000,
  ECO: 16000,
};

const rainVertexShader = /* glsl */ `
  attribute vec3 aOffset;
  attribute float aSpeed;
  attribute float aPhase;
  attribute float aLength;
  attribute float aGlow;
  uniform float uTime;
  uniform float uIntensity;
  uniform vec2 uWind;
  varying float vAlpha;
  varying float vGlow;

  void main() {
    float cycle = fract(uTime * aSpeed + aPhase);
    float y = mix(22.0, -0.12, cycle);
    float tail = position.y * aLength * (0.72 + uIntensity * 0.35);
    vec3 p = vec3(
      aOffset.x + uWind.x * cycle * 3.0,
      y + tail,
      aOffset.z + uWind.y * cycle * 2.0
    );
    float birth = smoothstep(0.0, 0.06, cycle);
    float impact = 1.0 - smoothstep(0.90, 1.0, cycle);
    float depthFade = smoothstep(-12.0, 12.0, -p.z);
    vAlpha = birth * impact * mix(0.16, 0.92, aGlow) * mix(0.45, 1.0, depthFade);
    vGlow = aGlow;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }
`;

const rainFragmentShader = /* glsl */ `
  varying float vAlpha;
  varying float vGlow;
  void main() {
    vec3 cold = vec3(0.20, 0.48, 0.78);
    vec3 white = vec3(0.82, 0.95, 1.0);
    vec3 color = mix(cold, white, vGlow);
    gl_FragColor = vec4(color, vAlpha);
  }
`;

const floorVertexShader = /* glsl */ `
  varying vec3 vWorld;
  void main() {
    vec4 world = modelMatrix * vec4(position, 1.0);
    vWorld = world.xyz;
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

const floorFragmentShader = /* glsl */ `
  uniform float uTime;
  uniform vec2 uMouse;
  varying vec3 vWorld;

  float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
  }

  float ring(vec2 p, vec2 c, float phase) {
    float d = length(p - c);
    float wave = abs(sin(d * 4.6 - phase * 6.0));
    return smoothstep(0.10, 0.0, wave) * smoothstep(8.0, 0.2, d);
  }

  void main() {
    vec2 p = vWorld.xz;
    vec2 cell = floor(p * 0.75);
    float rnd = hash(cell);
    float localTime = fract(uTime * (0.22 + rnd * 0.12) + rnd);
    vec2 center = (cell + vec2(hash(cell + 1.7), hash(cell + 8.3))) / 0.75;
    float randomRipple = ring(p, center, localTime) * smoothstep(0.82, 1.0, localTime) * step(0.76, rnd);

    float pools = 0.0;
    pools += ring(p, vec2(-8.0, -3.0), uTime * 0.22);
    pools += ring(p, vec2(7.0, -12.0), uTime * 0.19);
    pools += ring(p, vec2(1.0, 5.0), uTime * 0.24);

    vec2 grid = abs(fract(p * 0.25) - 0.5);
    float seam = smoothstep(0.025, 0.0, min(grid.x, grid.y));
    float sheen = pow(max(0.0, 1.0 - length(p - vec2(uMouse.x * 8.0, uMouse.y * 5.0)) / 13.0), 3.0);
    float horizon = smoothstep(23.0, -12.0, p.y);

    vec3 base = vec3(0.004, 0.010, 0.016);
    vec3 blue = vec3(0.08, 0.30, 0.52);
    vec3 color = base + blue * (randomRipple * 0.34 + pools * 0.18 + sheen * 0.10);
    color += vec3(0.035, 0.055, 0.07) * seam;
    color += vec3(0.02, 0.04, 0.06) * (0.5 + 0.5 * sin(p.x * 0.18 + uTime * 0.12)) * horizon;
    gl_FragColor = vec4(color, 1.0);
  }
`;

const impactVertexShader = /* glsl */ `
  attribute vec3 aOffset;
  attribute float aPhase;
  attribute float aSize;
  uniform float uTime;
  uniform float uPixelRatio;
  varying float vLife;

  void main() {
    float life = fract(uTime * 0.36 + aPhase);
    float pulse = smoothstep(0.86, 0.94, life) * (1.0 - smoothstep(0.94, 1.0, life));
    vec3 p = vec3(aOffset.x, 0.025, aOffset.z);
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = min(22.0, aSize * pulse * uPixelRatio * (180.0 / -mv.z));
    vLife = pulse;
  }
`;

const impactFragmentShader = /* glsl */ `
  varying float vLife;
  void main() {
    vec2 p = gl_PointCoord - 0.5;
    float d = length(p);
    float core = smoothstep(0.5, 0.0, d);
    float rim = smoothstep(0.07, 0.0, abs(d - 0.33));
    gl_FragColor = vec4(vec3(0.42, 0.76, 1.0) * (core + rim), (core * 0.5 + rim) * vLife);
  }
`;

function pickAutoProfile() {
  const memory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 4;
  const cores = navigator.hardwareConcurrency ?? 4;
  const mobile = /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent);
  if (!mobile && memory >= 8 && cores >= 12) return { profile: "ULTRA", drops: 70000 };
  if (!mobile && memory >= 6 && cores >= 8) return { profile: "HIGH", drops: 50000 };
  if (memory >= 4 && cores >= 4) return { profile: "BALANCED", drops: 32000 };
  return { profile: "ECO", drops: 16000 };
}

function makeRainGeometry(maxDrops: number) {
  const positions = new Float32Array(maxDrops * 2 * 3);
  const offsets = new Float32Array(maxDrops * 2 * 3);
  const speeds = new Float32Array(maxDrops * 2);
  const phases = new Float32Array(maxDrops * 2);
  const lengths = new Float32Array(maxDrops * 2);
  const glows = new Float32Array(maxDrops * 2);

  for (let i = 0; i < maxDrops; i++) {
    const angle = Math.random() * Math.PI * 2;
    const radius = Math.pow(Math.random(), 0.7) * 26;
    const x = Math.cos(angle) * radius;
    const z = Math.sin(angle) * radius - 3;
    const speed = 0.19 + Math.random() * 0.34;
    const phase = Math.random();
    const length = 0.35 + Math.pow(Math.random(), 2.2) * 2.8;
    const glow = Math.pow(Math.random(), 2.2);
    for (let v = 0; v < 2; v++) {
      const p = (i * 2 + v) * 3;
      positions[p + 1] = -v;
      offsets[p] = x;
      offsets[p + 2] = z;
      const a = i * 2 + v;
      speeds[a] = speed;
      phases[a] = phase;
      lengths[a] = length;
      glows[a] = glow;
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("aOffset", new THREE.BufferAttribute(offsets, 3));
  geometry.setAttribute("aSpeed", new THREE.BufferAttribute(speeds, 1));
  geometry.setAttribute("aPhase", new THREE.BufferAttribute(phases, 1));
  geometry.setAttribute("aLength", new THREE.BufferAttribute(lengths, 1));
  geometry.setAttribute("aGlow", new THREE.BufferAttribute(glows, 1));
  return geometry;
}

function makeImpactGeometry(count: number) {
  const positions = new Float32Array(count * 3);
  const offsets = new Float32Array(count * 3);
  const phases = new Float32Array(count);
  const sizes = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const angle = Math.random() * Math.PI * 2;
    const radius = Math.pow(Math.random(), 0.65) * 24;
    offsets[i * 3] = Math.cos(angle) * radius;
    offsets[i * 3 + 2] = Math.sin(angle) * radius - 3;
    phases[i] = Math.random();
    sizes[i] = 1.2 + Math.random() * 4.6;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("aOffset", new THREE.BufferAttribute(offsets, 3));
  geometry.setAttribute("aPhase", new THREE.BufferAttribute(phases, 1));
  geometry.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
  return geometry;
}

function qualityLabel(value: Quality) {
  return value === "AUTO" ? "自动适配" : value;
}

export default function RainExperience() {
  const canvasHost = useRef<HTMLDivElement>(null);
  const intensityRef = useRef(1);
  const cinematicRef = useRef(true);
  const qualityRef = useRef<Quality>("AUTO");
  const [intensity, setIntensity] = useState(100);
  const [quality, setQuality] = useState<Quality>("AUTO");
  const [cinematic, setCinematic] = useState(true);
  const [telemetry, setTelemetry] = useState<Telemetry>({
    fps: 60,
    drops: 0,
    profile: "检测中",
    gpu: "正在读取 GPU",
    adaptive: true,
  });

  useEffect(() => {
    intensityRef.current = intensity / 100;
  }, [intensity]);

  useEffect(() => {
    qualityRef.current = quality;
  }, [quality]);

  useEffect(() => {
    cinematicRef.current = cinematic;
  }, [cinematic]);

  useEffect(() => {
    const host = canvasHost.current;
    if (!host) return;

    const maxDrops = 70000;
    const auto = pickAutoProfile();
    let currentDrops = auto.drops;
    let targetDrops = auto.drops;
    let avgFps = 60;
    let frameCount = 0;
    let sampleStart = performance.now();
    let lastTime = performance.now();
    let raf = 0;
    let disposed = false;
    const mouse = new THREE.Vector2();
    const smoothMouse = new THREE.Vector2();
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const renderer = new THREE.WebGLRenderer({
      antialias: false,
      alpha: false,
      powerPreference: "high-performance",
      logarithmicDepthBuffer: false,
    });
    renderer.setClearColor(0x010305, 1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.82;
    const initialDpr = Math.min(window.devicePixelRatio, auto.profile === "ECO" ? 1 : auto.profile === "BALANCED" ? 1.35 : 1.75);
    renderer.setPixelRatio(initialDpr);
    renderer.setSize(host.clientWidth, host.clientHeight);
    host.appendChild(renderer.domElement);

    const gl = renderer.getContext();
    const debugInfo = gl.getExtension("WEBGL_debug_renderer_info");
    const gpu = debugInfo
      ? String(gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL))
      : String(gl.getParameter(gl.RENDERER));
    const gpuShort = gpu.replace(/ANGLE \(|Direct3D.*$/g, "").replace(/\s+/g, " ").trim().slice(0, 42);

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x01050a, 0.026);
    const camera = new THREE.PerspectiveCamera(48, host.clientWidth / host.clientHeight, 0.1, 120);
    camera.position.set(13, 6.4, 18);

    const composer = new EffectComposer(renderer);
    composer.setPixelRatio(initialDpr);
    composer.addPass(new RenderPass(scene, camera));
    const bloom = new UnrealBloomPass(
      new THREE.Vector2(host.clientWidth, host.clientHeight),
      auto.profile === "ECO" ? 0.65 : 1.05,
      0.72,
      0.12,
    );
    composer.addPass(bloom);

    const rainUniforms = {
      uTime: { value: 0 },
      uIntensity: { value: 1 },
      uWind: { value: new THREE.Vector2(0.11, -0.03) },
    };
    const rainGeometry = makeRainGeometry(maxDrops);
    rainGeometry.setDrawRange(0, currentDrops * 2);
    const rainMaterial = new THREE.ShaderMaterial({
      vertexShader: rainVertexShader,
      fragmentShader: rainFragmentShader,
      uniforms: rainUniforms,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const rain = new THREE.LineSegments(rainGeometry, rainMaterial);
    scene.add(rain);

    const floorUniforms = {
      uTime: { value: 0 },
      uMouse: { value: new THREE.Vector2() },
    };
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(72, 72, 1, 1),
      new THREE.ShaderMaterial({
        vertexShader: floorVertexShader,
        fragmentShader: floorFragmentShader,
        uniforms: floorUniforms,
        side: THREE.DoubleSide,
      }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.14;
    scene.add(floor);

    const impactUniforms = {
      uTime: { value: 0 },
      uPixelRatio: { value: initialDpr },
    };
    const impacts = new THREE.Points(
      makeImpactGeometry(4200),
      new THREE.ShaderMaterial({
        vertexShader: impactVertexShader,
        fragmentShader: impactFragmentShader,
        uniforms: impactUniforms,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    scene.add(impacts);

    const structureMaterial = new THREE.MeshStandardMaterial({
      color: 0x020508,
      roughness: 0.22,
      metalness: 0.72,
    });
    const pillarGeometry = new THREE.BoxGeometry(0.12, 8, 0.12);
    const pillars = new THREE.InstancedMesh(pillarGeometry, structureMaterial, 72);
    const matrix = new THREE.Matrix4();
    for (let i = 0; i < 72; i++) {
      const side = i % 2 === 0 ? -1 : 1;
      const row = Math.floor(i / 2);
      const x = side * (10 + Math.sin(row * 0.62) * 2.4);
      const z = 10 - row * 0.78;
      const height = 2.5 + Math.random() * 7;
      matrix.compose(
        new THREE.Vector3(x, height / 2 - 0.1, z),
        new THREE.Quaternion(),
        new THREE.Vector3(1, height / 8, 1),
      );
      pillars.setMatrixAt(i, matrix);
    }
    scene.add(pillars);

    const ambient = new THREE.AmbientLight(0x1d3e55, 0.18);
    scene.add(ambient);
    const spots: THREE.PointLight[] = [];
    [
      [-8, 0.15, -3],
      [7, 0.15, -12],
      [1, 0.15, 5],
      [-1, 0.15, -18],
    ].forEach(([x, y, z], index) => {
      const light = new THREE.PointLight(index === 1 ? 0x9fdcff : 0x3e91cf, 13, 12, 2.25);
      light.position.set(x, y, z);
      scene.add(light);
      spots.push(light);
      const disc = new THREE.Mesh(
        new THREE.CircleGeometry(3.8, 48),
        new THREE.MeshBasicMaterial({
          color: index === 1 ? 0x7ec8f5 : 0x1d74aa,
          transparent: true,
          opacity: 0.06,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
        }),
      );
      disc.rotation.x = -Math.PI / 2;
      disc.position.set(x, 0.01, z);
      scene.add(disc);
    });

    const onPointerMove = (event: PointerEvent) => {
      const rect = host.getBoundingClientRect();
      mouse.set(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      );
    };

    const onResize = () => {
      const width = host.clientWidth;
      const height = host.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
      composer.setSize(width, height);
    };

    host.addEventListener("pointermove", onPointerMove);
    window.addEventListener("resize", onResize);

    const clock = new THREE.Clock();
    const animate = () => {
      if (disposed) return;
      raf = requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();
      const now = performance.now();
      const delta = Math.min(0.1, (now - lastTime) / 1000);
      lastTime = now;

      smoothMouse.lerp(mouse, 1 - Math.pow(0.001, delta));
      rainUniforms.uTime.value = elapsed;
      rainUniforms.uIntensity.value = intensityRef.current;
      rainUniforms.uWind.value.x = 0.08 + Math.sin(elapsed * 0.11) * 0.08;
      floorUniforms.uTime.value = elapsed;
      floorUniforms.uMouse.value.copy(smoothMouse);
      impactUniforms.uTime.value = elapsed;

      const manual = qualityRef.current;
      if (manual !== "AUTO") {
        targetDrops = Math.round(qualityDrops[manual] * intensityRef.current);
      }
      targetDrops = Math.max(7000, Math.min(maxDrops, targetDrops));
      currentDrops += (targetDrops - currentDrops) * Math.min(1, delta * 1.8);
      rainGeometry.setDrawRange(0, Math.floor(currentDrops) * 2);

      if (cinematicRef.current && !reducedMotion) {
        const orbit = elapsed * 0.055;
        camera.position.x = 12.8 + Math.sin(orbit) * 7.5 + smoothMouse.x * 1.2;
        camera.position.z = 17.5 + Math.cos(orbit * 0.82) * 4.8;
        camera.position.y = 5.8 + Math.sin(orbit * 1.2) * 1.25 + smoothMouse.y * 0.55;
      } else {
        camera.position.x += ((12.5 + smoothMouse.x * 2.2) - camera.position.x) * 0.025;
        camera.position.y += ((6.0 + smoothMouse.y) - camera.position.y) * 0.025;
        camera.position.z += (18 - camera.position.z) * 0.025;
      }
      camera.lookAt(smoothMouse.x * 2.6, 1.0 + smoothMouse.y * 0.7, -5.5);

      spots.forEach((light, i) => {
        light.intensity = 10 + Math.sin(elapsed * (0.7 + i * 0.11) + i) * 3.5;
      });

      frameCount++;
      if (now - sampleStart > 1800) {
        avgFps = Math.round((frameCount * 1000) / (now - sampleStart));
        if (qualityRef.current === "AUTO") {
          if (avgFps < 42) targetDrops = Math.max(10000, targetDrops * 0.82);
          else if (avgFps > 57) targetDrops = Math.min(maxDrops, targetDrops * 1.09);
        }
        setTelemetry({
          fps: avgFps,
          drops: Math.round(currentDrops / 1000) * 1000,
          profile: qualityRef.current === "AUTO" ? auto.profile : qualityRef.current,
          gpu: gpuShort || "WebGL GPU",
          adaptive: qualityRef.current === "AUTO",
        });
        frameCount = 0;
        sampleStart = now;
      }

      composer.render();
    };

    setTelemetry({
      fps: 60,
      drops: currentDrops,
      profile: auto.profile,
      gpu: gpuShort || "WebGL GPU",
      adaptive: true,
    });
    animate();

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      host.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("resize", onResize);
      rainGeometry.dispose();
      rainMaterial.dispose();
      floor.geometry.dispose();
      (floor.material as THREE.Material).dispose();
      impacts.geometry.dispose();
      (impacts.material as THREE.Material).dispose();
      pillarGeometry.dispose();
      structureMaterial.dispose();
      composer.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  const cycleQuality = () => {
    const order: Quality[] = ["AUTO", "ULTRA", "HIGH", "BALANCED", "ECO"];
    setQuality(order[(order.indexOf(quality) + 1) % order.length]);
  };

  return (
    <main className="rain-app">
      <div ref={canvasHost} className="rain-canvas" aria-hidden="true" />
      <div className="vignette" aria-hidden="true" />
      <div className="scanlines" aria-hidden="true" />

      <header className="topbar">
        <a className="brand" href="#experience" aria-label="Data Rain 首页">
          <span className="brand-mark"><i /><i /><i /></span>
          DATA<span>RAIN</span>
        </a>
        <div className="top-status">
          <span className="live-dot" />
          LIVE SIMULATION
          <b>{telemetry.fps} FPS</b>
        </div>
      </header>

      <section id="experience" className="hero-copy">
        <p className="eyebrow">PRECIPITATION / 07.23 / SHANGHAI</p>
        <h1>降雨<br /><em>强度</em></h1>
        <p className="subtitle">每小时降水</p>
        <div className="reading">
          <strong>{(12.8 * intensity / 100).toFixed(1)}</strong>
          <span>毫米／小时</span>
        </div>
        <div className="axis" aria-hidden="true">
          <span>12.8</span>
          <i />
          <small>0.0</small>
        </div>
      </section>

      <aside className="telemetry" aria-label="实时渲染数据">
        <div className="telemetry-head">
          <span>系统遥测</span>
          <b>RENDER_01</b>
        </div>
        <dl>
          <div><dt>粒子数量</dt><dd>{telemetry.drops.toLocaleString("zh-CN")}</dd></div>
          <div><dt>当前帧率</dt><dd>{telemetry.fps} <small>FPS</small></dd></div>
          <div><dt>画质档位</dt><dd>{telemetry.profile}</dd></div>
          <div><dt>动态调节</dt><dd className={telemetry.adaptive ? "on" : ""}>{telemetry.adaptive ? "ACTIVE" : "MANUAL"}</dd></div>
        </dl>
        <p className="gpu-name">{telemetry.gpu}</p>
      </aside>

      <section className="controls" aria-label="场景控制">
        <label>
          <span>降雨强度</span>
          <input
            type="range"
            min="35"
            max="140"
            value={intensity}
            onChange={(event) => setIntensity(Number(event.target.value))}
            aria-label="调整降雨强度"
          />
          <b>{intensity}%</b>
        </label>
        <button type="button" onClick={cycleQuality}>
          <span>画质</span>
          <b>{qualityLabel(quality)}</b>
        </button>
        <button type="button" onClick={() => setCinematic((value) => !value)}>
          <span>镜头</span>
          <b>{cinematic ? "巡航中" : "跟随鼠标"}</b>
        </button>
      </section>

      <div className="scroll-cue" aria-hidden="true">
        <i />
        MOVE TO BEND THE STORM
      </div>
    </main>
  );
}
