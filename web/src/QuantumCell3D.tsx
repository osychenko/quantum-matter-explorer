/** Patch 34: explanatory 3-D visualization, deliberately not a trajectory simulator.
 *
 * The N=1 ideal host position is sampled uniformly (exact canonical one-body
 * position marginal). The optional two-particle paths are HAND-DRAWN SCHEMATA,
 * never presented as Monte Carlo output or as real-time dynamics. In the
 * recorded M=8 view all bead positions are illustrative only.
 */
import {useEffect, useMemo, useRef, useState} from 'react';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

export type QuantumView = 'configuration' | 'worldlines';
export type QuantumSceneKind = 'ideal' | 'recorded';
type Props = {length: number; kind: QuantumSceneKind; view: QuantumView; showImages: boolean;
              seed: number};

// Stable pure PRNG. No user claim that the positions depict measured records.
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a += 0x6D2B79F5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function idealHostPosition(seed: number): [number, number, number] {
  const random = mulberry32(seed);
  return [random(), random(), random()];
}

function sphere(color: number, radius = 0.046, opacity = 1): THREE.Mesh {
  const m = new THREE.Mesh(new THREE.SphereGeometry(radius, 24, 16),
    new THREE.MeshStandardMaterial({color, roughness: 0.28, metalness: 0.15,
      transparent: opacity < 1, opacity}));
  return m;
}

function pathTube(points: THREE.Vector3[], color: number, opacity = 1): THREE.Mesh {
  const mesh = new THREE.Mesh(
    new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points), 100, 0.012, 8, false),
    new THREE.MeshStandardMaterial({color, emissive: color, emissiveIntensity: 0.17,
      transparent: opacity < 1, opacity, roughness: 0.3}));
  return mesh;
}

// This is solely a graphic illustration of the identity vs exchanged
// permutations. It does not sample a path-integral measure.
function illustrativePaths(size: number): THREE.Group {
  const group = new THREE.Group();
  const tracks: [number, number, (t: number) => number][] = [
    [0x7af2e0, -1, t => -0.22 + 0.45 * t],
    [0xe5ad75, 1, t => 0.22 - 0.45 * t],
  ];
  for (const [color, sign, xPath] of tracks) {
    const points: THREE.Vector3[] = [];
    for (let i = 0; i <= 60; i++) {
      const t = i / 60;
      points.push(new THREE.Vector3(
        xPath(t) * size + 0.07 * Math.sin(2 * Math.PI * t) * size,
        (t - 0.5) * 0.90 * size,
        sign * 0.16 * size + 0.06 * Math.sin(Math.PI * t) * size));
    }
    group.add(pathTube(points, color));
    for (const end of [points[0], points[points.length - 1]]) {
      const bead = sphere(color, 0.030 * size);
      bead.position.copy(end);
      group.add(bead);
    }
  }
  return group;
}

export default function QuantumCell3D({length, kind, view, showImages, seed}: Props) {
  const mount = useRef<HTMLDivElement>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const [error, setError] = useState<string | null>(null);
  const normalized = useMemo(() => idealHostPosition(seed), [seed]);

  useEffect(() => {
    const host = mount.current;
    if (!host) return;
    let animationFrame = 0;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
    } catch {
      setError('3D graphics are unavailable in this browser. The numerical results remain accessible above.');
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(Math.max(1, host.clientWidth), Math.max(1, host.clientHeight));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100);
    camera.position.set(2.5, 1.85, 2.8);
    camera.lookAt(0, 0, 0);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.075;
    controls.enablePan = false;
    controls.minDistance = 1.4;
    controls.maxDistance = 7;
    controlsRef.current = controls;
    scene.add(new THREE.AmbientLight(0xeaf9ff, 1.5));
    const light = new THREE.DirectionalLight(0xa8f3e7, 2.2);
    light.position.set(2, 3, 4);
    scene.add(light);
    const backLight = new THREE.DirectionalLight(0xf2b678, 0.8);
    backLight.position.set(-2, -1, -3);
    scene.add(backLight);

    // Display a unit cell. Physical L is represented by axis label, not
    // by misleadingly rescaling the on-screen camera in response to L.
    const box = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
      new THREE.LineBasicMaterial({color: 0x6a98ac, transparent: true, opacity: 0.88}));
    scene.add(box);
    const inner = new THREE.GridHelper(1, 6, 0x28546b, 0x224156);
    inner.position.y = -0.5;
    scene.add(inner);
    if (view === 'configuration') {
      const position = new THREE.Vector3(normalized[0] - 0.5, normalized[1] - 0.5, normalized[2] - 0.5);
      const hostParticle = sphere(0x7af2e0);
      hostParticle.position.copy(position);
      scene.add(hostParticle);
      if (kind === 'recorded') {
        // Explicitly a demonstration glyph, not an observed configuration.
        const second = sphere(0xe5ad75);
        second.position.set(-position.x * .65 + .10, position.y * .20 - .12, -position.z * .70);
        scene.add(second);
      }
      if (showImages) {
        for (const dx of [-1, 0, 1]) for (const dy of [-1, 0, 1]) for (const dz of [-1, 0, 1]) {
          if (dx === 0 && dy === 0 && dz === 0) continue;
          if (Math.abs(dx) + Math.abs(dy) + Math.abs(dz) > 1) continue;
          const ghost = sphere(0x7af2e0, .044, .21);
          ghost.position.copy(position).add(new THREE.Vector3(dx, dy, dz));
          scene.add(ghost);
        }
      }
    } else {
      scene.add(illustrativePaths(1));
    }
    const resize = () => {
      const width = Math.max(1, host.clientWidth), height = Math.max(1, host.clientHeight);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();
    const loop = () => {controls.update(); renderer.render(scene, camera); animationFrame = requestAnimationFrame(loop);};
    loop();
    return () => {
      cancelAnimationFrame(animationFrame);
      observer.disconnect();
      controls.dispose();
      controlsRef.current = null;
      scene.traverse(obj => {
        const mesh = obj as THREE.Mesh;
        if (mesh.geometry) mesh.geometry.dispose();
        if (mesh.material) {
          for (const mat of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) mat.dispose();
        }
      });
      renderer.dispose();
      if (host.contains(renderer.domElement)) host.removeChild(renderer.domElement);
    };
  }, [normalized, view, kind, showImages]);

  return <div className="three-container">
    <div ref={mount} className="three-viewport" role="img" aria-label={view === 'configuration' ?
      `3D periodic cell with ${kind === 'ideal' ? 'an ideal one-particle uniform position draw' : 'illustrative particle positions'}` :
      'Schematic imaginary-time exchange paths; not real-time motion or a sampled Monte Carlo trajectory'} />
    {error && <p role="status" className="three-error">{error}</p>}
    <div className="three-overlay"><span>3D PERIODIC CELL</span><strong>L = {length.toFixed(2)}</strong></div>
    <button className="three-reset" type="button" onClick={() => controlsRef.current?.reset()}>↺ Reset view</button>
  </div>;
}
