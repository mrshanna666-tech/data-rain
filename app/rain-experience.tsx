"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";

const initialRainfall = [
  2.1, 3.8, 9.9, 11.9, 0.0, 1.4, 0.7, 0.6, 4.1, 7.8, 6.6, 2.4,
  1.4, 1.6, 2.2, 3.8, 5.9, 7.4, 10.0, 8.8, 5.3, 3.2, 1.2, 0.4,
];

type Telemetry = {
  fps: number;
  drops: number;
  profile: string;
};

const rainVertexShader = /* glsl */ `
  attribute vec3 aOffset;
  attribute float aSpeed;
  attribute float aPhase;
  attribute float aLength;
  attribute float aGlow;
  attribute float aHeight;
  uniform float uTime;
  varying float vAlpha;
  varying float vGlow;

  void main() {
    float cycle = fract(uTime * aSpeed + aPhase);
    float y = mix(aHeight + 0.45, 0.03, cycle);
    vec3 p = vec3(aOffset.x, y + position.y * aLength, aOffset.z);
    float born = smoothstep(0.0, 0.08, cycle);
    float impact = 1.0 - smoothstep(0.88, 1.0, cycle);
    float depth = smoothstep(-2.6, 2.2, p.z);
    vAlpha = born * impact * mix(0.055, 0.32, aGlow) * mix(0.52, 1.0, depth);
    vGlow = aGlow;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }
`;

const rainFragmentShader = /* glsl */ `
  varying float vAlpha;
  varying float vGlow;
  void main() {
    vec3 graphite = vec3(0.19, 0.22, 0.24);
    vec3 silver = vec3(0.52, 0.60, 0.64);
    gl_FragColor = vec4(mix(graphite, silver, vGlow), vAlpha);
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
  varying vec3 vWorld;

  float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
  }

  void main() {
    vec2 p = vWorld.xz;
    float strip = smoothstep(3.8, 2.7, abs(p.y));
    vec2 cell = floor(vec2((p.x + 13.0) * 1.4, (p.y + 3.0) * 1.7));
    float rnd = hash(cell);
    float life = fract(uTime * (0.18 + rnd * 0.08) + rnd);
    vec2 center = vec2(
      (cell.x + hash(cell + 2.4)) / 1.4 - 13.0,
      (cell.y + hash(cell + 7.1)) / 1.7 - 3.0
    );
    float d = length(p - center);
    float ring = smoothstep(0.035, 0.0, abs(d - life * 0.34))
      * (1.0 - life) * step(0.79, rnd) * strip;
    float wet = strip * (0.35 + 0.65 * hash(floor(p * vec2(0.38, 1.8))));
    vec3 base = vec3(0.0005, 0.0010, 0.0014);
    vec3 color = base + vec3(0.018, 0.024, 0.028) * wet;
    color += vec3(0.075, 0.105, 0.12) * ring;
    gl_FragColor = vec4(color, 1.0);
  }
`;

const impactVertexShader = /* glsl */ `
  attribute vec3 aOffset;
  attribute float aPhase;
  attribute float aSize;
  uniform float uTime;
  uniform float uPixelRatio;
  varying float vPulse;

  void main() {
    float life = fract(uTime * 0.31 + aPhase);
    float pulse = smoothstep(0.89, 0.945, life) * (1.0 - smoothstep(0.945, 1.0, life));
    vec4 mv = modelViewMatrix * vec4(aOffset.x, 0.024, aOffset.z, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = min(8.0, aSize * pulse * uPixelRatio * (95.0 / -mv.z));
    vPulse = pulse;
  }
`;

const impactFragmentShader = /* glsl */ `
  varying float vPulse;
  void main() {
    vec2 p = gl_PointCoord - 0.5;
    float d = length(p);
    float core = smoothstep(0.44, 0.0, d);
    float rim = smoothstep(0.035, 0.0, abs(d - 0.31));
    gl_FragColor = vec4(vec3(0.30, 0.39, 0.43), (core * 0.12 + rim * 0.28) * vPulse);
  }
`;

function weightedHour(data: number[], total: number) {
  let marker = Math.random() * total;
  for (let i = 0; i < data.length; i++) {
    marker -= Math.max(0.08, data[i]);
    if (marker <= 0) return i;
  }
  return data.length - 1;
}

function makeRainGeometry(maxDrops: number, data: number[]) {
  const positions = new Float32Array(maxDrops * 2 * 3);
  const offsets = new Float32Array(maxDrops * 2 * 3);
  const speeds = new Float32Array(maxDrops * 2);
  const phases = new Float32Array(maxDrops * 2);
  const lengths = new Float32Array(maxDrops * 2);
  const glows = new Float32Array(maxDrops * 2);
  const heights = new Float32Array(maxDrops * 2);
  const total = data.reduce((sum, value) => sum + Math.max(0.08, value), 0);

  for (let i = 0; i < maxDrops; i++) {
    const hour = weightedHour(data, total);
    const x = -12.2 + (hour / (data.length - 1)) * 24.4 + (Math.random() - 0.5) * 0.82;
    const z = (Math.random() - 0.5) * 4.6;
    const height = 0.18 + (data[hour] / 12) * 8.4;
    const speed = 0.16 + Math.random() * 0.24;
    const phase = Math.random();
    const length = 0.18 + Math.pow(Math.random(), 2.4) * 1.25;
    const glow = Math.pow(Math.random(), 3.4);

    for (let vertex = 0; vertex < 2; vertex++) {
      const index = i * 2 + vertex;
      const p = index * 3;
      positions[p + 1] = -vertex;
      offsets[p] = x;
      offsets[p + 2] = z;
      speeds[index] = speed;
      phases[index] = phase;
      lengths[index] = length;
      glows[index] = glow;
      heights[index] = height;
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("aOffset", new THREE.BufferAttribute(offsets, 3));
  geometry.setAttribute("aSpeed", new THREE.BufferAttribute(speeds, 1));
  geometry.setAttribute("aPhase", new THREE.BufferAttribute(phases, 1));
  geometry.setAttribute("aLength", new THREE.BufferAttribute(lengths, 1));
  geometry.setAttribute("aGlow", new THREE.BufferAttribute(glows, 1));
  geometry.setAttribute("aHeight", new THREE.BufferAttribute(heights, 1));
  return geometry;
}

function makeImpactGeometry(count: number, data: number[]) {
  const positions = new Float32Array(count * 3);
  const offsets = new Float32Array(count * 3);
  const phases = new Float32Array(count);
  const sizes = new Float32Array(count);
  const total = data.reduce((sum, value) => sum + Math.max(0.08, value), 0);

  for (let i = 0; i < count; i++) {
    const hour = weightedHour(data, total);
    offsets[i * 3] = -12.2 + (hour / (data.length - 1)) * 24.4 + (Math.random() - 0.5) * 0.8;
    offsets[i * 3 + 2] = (Math.random() - 0.5) * 4.5;
    phases[i] = Math.random();
    sizes[i] = 1.0 + Math.random() * 2.4;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("aOffset", new THREE.BufferAttribute(offsets, 3));
  geometry.setAttribute("aPhase", new THREE.BufferAttribute(phases, 1));
  geometry.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
  return geometry;
}

function makeAxis(data: number[]) {
  const group = new THREE.Group();
  const lineMaterial = new THREE.LineBasicMaterial({
    color: 0xaab9c1,
    transparent: true,
    opacity: 0.23,
  });
  const faintMaterial = new THREE.LineBasicMaterial({
    color: 0x8fa8b4,
    transparent: true,
    opacity: 0.075,
  });

  const addLine = (points: THREE.Vector3[], material = lineMaterial) => {
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    group.add(new THREE.Line(geometry, material));
  };

  addLine([new THREE.Vector3(-13.1, 0, 2.7), new THREE.Vector3(13.4, 0, 2.7)]);
  addLine([new THREE.Vector3(-13.1, 0, 2.7), new THREE.Vector3(-13.1, 9.1, 2.7)]);

  for (let i = 0; i <= 6; i++) {
    const x = -12.2 + (i / 6) * 24.4;
    addLine([new THREE.Vector3(x, 0, 2.66), new THREE.Vector3(x, 0, 2.88)], faintMaterial);
  }
  for (let i = 1; i <= 3; i++) {
    const y = i * 2.8;
    addLine([new THREE.Vector3(-13.1, y, 2.65), new THREE.Vector3(-12.85, y, 2.65)], faintMaterial);
  }

  const curvePoints = data.map((value, index) =>
    new THREE.Vector3(
      -12.2 + (index / (data.length - 1)) * 24.4,
      0.16 + (value / 12) * 8.4,
      0,
    ),
  );
  const curve = new THREE.CatmullRomCurve3(curvePoints, false, "catmullrom", 0.08);
  addLine(curve.getPoints(180), new THREE.LineBasicMaterial({
    color: 0xc2d1d8,
    transparent: true,
    opacity: 0.12,
  }));
  return group;
}

function selectProfile() {
  const memory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 4;
  const cores = navigator.hardwareConcurrency ?? 4;
  const mobile = /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent);
  if (!mobile && memory >= 8 && cores >= 12) return { name: "ULTRA", drops: 52000, dpr: 1.55 };
  if (!mobile && memory >= 6 && cores >= 8) return { name: "HIGH", drops: 40000, dpr: 1.35 };
  if (memory >= 4 && cores >= 4) return { name: "BALANCED", drops: 28000, dpr: 1.15 };
  return { name: "ECO", drops: 14000, dpr: 1 };
}

export default function RainExperience() {
  const hostRef = useRef<HTMLDivElement>(null);
  const [rainfall, setRainfall] = useState(initialRainfall);
  const [draft, setDraft] = useState(initialRainfall);
  const [editorOpen, setEditorOpen] = useState(false);
  const [telemetry, setTelemetry] = useState<Telemetry>({ fps: 60, drops: 0, profile: "检测中" });
  const peak = useMemo(() => Math.max(...rainfall), [rainfall]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const profile = selectProfile();
    const maxDrops = 56000;
    let targetDrops = profile.drops;
    let currentDrops = profile.drops;
    let frameCount = 0;
    let sampleStart = performance.now();
    let raf = 0;
    let disposed = false;
    const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const pointer = new THREE.Vector2();
    const pointerSmooth = new THREE.Vector2();

    const renderer = new THREE.WebGLRenderer({
      antialias: false,
      powerPreference: "high-performance",
    });
    const dpr = Math.min(window.devicePixelRatio, profile.dpr);
    renderer.setPixelRatio(dpr);
    renderer.setSize(host.clientWidth, host.clientHeight);
    renderer.setClearColor(0x000101, 1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.28;
    host.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x000101, 0.038);
    const camera = new THREE.PerspectiveCamera(42, host.clientWidth / host.clientHeight, 0.1, 100);
    camera.position.set(15.8, 6.6, 17.8);

    const composer = new EffectComposer(renderer);
    composer.setPixelRatio(dpr);
    composer.addPass(new RenderPass(scene, camera));
    const bloom = new UnrealBloomPass(
      new THREE.Vector2(host.clientWidth, host.clientHeight),
      0.14,
      0.28,
      0.91,
    );
    composer.addPass(bloom);

    const rainUniforms = { uTime: { value: 0 } };
    const rainGeometry = makeRainGeometry(maxDrops, rainfall);
    rainGeometry.setDrawRange(0, currentDrops * 2);
    const rainMaterial = new THREE.ShaderMaterial({
      vertexShader: rainVertexShader,
      fragmentShader: rainFragmentShader,
      uniforms: rainUniforms,
      transparent: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
    });
    scene.add(new THREE.LineSegments(rainGeometry, rainMaterial));

    const floorUniforms = { uTime: { value: 0 } };
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(48, 22),
      new THREE.ShaderMaterial({
        vertexShader: floorVertexShader,
        fragmentShader: floorFragmentShader,
        uniforms: floorUniforms,
      }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.035;
    scene.add(floor);

    const impactUniforms = {
      uTime: { value: 0 },
      uPixelRatio: { value: dpr },
    };
    const impacts = new THREE.Points(
      makeImpactGeometry(2600, rainfall),
      new THREE.ShaderMaterial({
        vertexShader: impactVertexShader,
        fragmentShader: impactFragmentShader,
        uniforms: impactUniforms,
        transparent: true,
        depthWrite: false,
        blending: THREE.NormalBlending,
      }),
    );
    scene.add(impacts);
    const axis = makeAxis(rainfall);
    scene.add(axis);

    const onPointer = (event: PointerEvent) => {
      const rect = host.getBoundingClientRect();
      pointer.set(
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
    host.addEventListener("pointermove", onPointer);
    window.addEventListener("resize", onResize);

    const clock = new THREE.Clock();
    const animate = () => {
      if (disposed) return;
      raf = requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();
      pointerSmooth.lerp(pointer, 0.025);
      rainUniforms.uTime.value = elapsed;
      floorUniforms.uTime.value = elapsed;
      impactUniforms.uTime.value = elapsed;

      currentDrops += (targetDrops - currentDrops) * 0.025;
      rainGeometry.setDrawRange(0, Math.floor(currentDrops) * 2);

      const drift = reducedMotion ? 0 : Math.sin(elapsed * 0.045) * 1.6;
      camera.position.x = 15.8 + drift + pointerSmooth.x * 0.45;
      camera.position.y = 6.6 + Math.cos(elapsed * 0.038) * 0.35 + pointerSmooth.y * 0.22;
      camera.position.z = 17.8 + Math.sin(elapsed * 0.032) * 0.8;
      camera.lookAt(-0.8 + drift * 0.12, 2.2, -0.4);

      frameCount++;
      const now = performance.now();
      if (now - sampleStart > 1900) {
        const fps = Math.round((frameCount * 1000) / (now - sampleStart));
        if (fps < 43) targetDrops = Math.max(10000, targetDrops * 0.82);
        if (fps > 57) targetDrops = Math.min(maxDrops, targetDrops * 1.07);
        setTelemetry({
          fps,
          drops: Math.round(currentDrops / 1000) * 1000,
          profile: profile.name,
        });
        frameCount = 0;
        sampleStart = now;
      }
      composer.render();
    };

    setTelemetry({ fps: 60, drops: profile.drops, profile: profile.name });
    animate();

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      host.removeEventListener("pointermove", onPointer);
      window.removeEventListener("resize", onResize);
      rainGeometry.dispose();
      rainMaterial.dispose();
      floor.geometry.dispose();
      (floor.material as THREE.Material).dispose();
      impacts.geometry.dispose();
      (impacts.material as THREE.Material).dispose();
      axis.traverse((child) => {
        if (child instanceof THREE.Line) {
          child.geometry.dispose();
          child.material.dispose();
        }
      });
      composer.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [rainfall]);

  const applyData = () => {
    setRainfall(draft.map((value) => Math.max(0, Math.min(20, Number(value) || 0))));
    setEditorOpen(false);
  };

  return (
    <main className="rain-app">
      <div ref={hostRef} className="rain-canvas" aria-hidden="true" />
      <div className="vignette" aria-hidden="true" />

      <section className="hero-copy">
        <p className="eyebrow">PRECIPITATION / 24 HOURS</p>
        <h1>降雨强度</h1>
        <p className="subtitle">每小时降水</p>
        <div className="reading">
          <strong>{peak.toFixed(1)}</strong>
          <span>毫米／小时</span>
        </div>
        <div className="vertical-scale" aria-hidden="true">
          <span>{peak.toFixed(1)}</span>
          <i />
          <small>0</small>
        </div>
      </section>

      <div className="timeline" aria-hidden="true">
        {["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"].map((time) => (
          <span key={time}>{time}</span>
        ))}
      </div>

      <div className="render-status">
        <span>{telemetry.drops.toLocaleString("zh-CN")} 粒子</span>
        <i />
        <span>{telemetry.fps} FPS</span>
        <i />
        <span>{telemetry.profile} / AUTO</span>
      </div>

      <button
        type="button"
        className="edit-trigger"
        onClick={() => setEditorOpen(true)}
        aria-expanded={editorOpen}
      >
        编辑降雨数据
      </button>

      <aside className={`data-editor ${editorOpen ? "open" : ""}`} aria-hidden={!editorOpen}>
        <div className="editor-head">
          <div>
            <small>00:00—24:00</small>
            <h2>编辑降雨数据</h2>
          </div>
          <button type="button" onClick={() => setEditorOpen(false)} aria-label="关闭编辑器">×</button>
        </div>
        <p>输入各时段的毫米／小时数据，雨幕高度和密度会随之重新生成。</p>
        <div className="data-grid">
          {draft.map((value, index) => (
            <label key={index}>
              <span>{String(index).padStart(2, "0")}:00</span>
              <input
                type="number"
                min="0"
                max="20"
                step="0.1"
                value={value}
                onChange={(event) => {
                  const next = [...draft];
                  next[index] = Number(event.target.value);
                  setDraft(next);
                }}
              />
            </label>
          ))}
        </div>
        <button type="button" className="apply-data" onClick={applyData}>应用到雨幕</button>
      </aside>
    </main>
  );
}
