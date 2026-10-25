// OCTO-EARTH 3D rig - shared by the hydrogen-collecting-disk and dyson-sphere sims.
// Built in the style of frontend/app.js (three.js r160 from frontend/vendor, same material
// palette: Earth #4fa3ff / emissive #0a2a55, metal hands #e8f4ff). The repo's old viewer only
// had a plain sphere, so the eight octo-hands are modelled here once and reused:
//   body sphere + 8 two-segment arms (shoulder -> elbow -> wrist) ending in a hand.
//   hand = 'pad'    : small concave scoop (hydrogen net)
//   hand = 'mirror' : huge concave solar absorber, cave side facing the arm direction
// face = +1 : hands open toward local +X   (hydrogen: into the ram flow)
// face = -1 : hands open toward local -X   (dyson: cave side on the Sun side)
import * as THREE from '../frontend/vendor/three.module.js';

const Z = new THREE.Vector3(0, 0, 1);

export function buildOctoEarth({ hand = 'pad', bodyR = 1, mirrorR = 1.6, face = 1 } = {}) {
  const group = new THREE.Group();
  const bodyMat = new THREE.MeshStandardMaterial({ color: 0x4fa3ff, emissive: 0x0a2a55, roughness: .55, metalness: .1 });
  const body = new THREE.Mesh(new THREE.SphereGeometry(bodyR, 32, 24), bodyMat);
  group.add(body);

  const armMat = new THREE.MeshStandardMaterial({ color: 0xcfd8e3, metalness: .55, roughness: .35 });
  const jointMat = new THREE.MeshStandardMaterial({ color: 0xe8f4ff, metalness: .7, roughness: .25 });
  const handMat = new THREE.MeshStandardMaterial({ color: 0xe8f4ff, metalness: .92, roughness: .13, side: THREE.DoubleSide });
  const L1 = bodyR * 1.15, L2 = bodyR * 1.05;

  const arms = [];
  for (let k = 0; k < 8; k++) {
    const th = (k / 8) * Math.PI * 2;
    const dir = new THREE.Vector3(face * .45, Math.cos(th), Math.sin(th)).normalize();
    const shoulder = new THREE.Group();
    shoulder.position.copy(dir).multiplyScalar(bodyR * .92);
    shoulder.quaternion.setFromUnitVectors(Z, dir);          // local +Z runs along the arm
    const upper = new THREE.Mesh(new THREE.CylinderGeometry(.06, .09, L1, 10), armMat);
    upper.geometry.rotateX(Math.PI / 2); upper.position.z = L1 / 2;
    const elbow = new THREE.Mesh(new THREE.SphereGeometry(.105, 12, 8), jointMat);
    elbow.position.z = L1;
    const fore = new THREE.Group(); fore.position.z = L1;
    const foreM = new THREE.Mesh(new THREE.CylinderGeometry(.045, .062, L2, 10), armMat);
    foreM.geometry.rotateX(Math.PI / 2); foreM.position.z = L2 / 2;
    const wrist = new THREE.Mesh(new THREE.SphereGeometry(.082, 12, 8), jointMat);
    wrist.position.z = L2;
    fore.add(foreM, wrist);
    const handG = new THREE.Group(); handG.position.z = L2;
    let handMesh;
    if (hand === 'mirror') {
      // concave absorber: open spherical cap, cave side toward local +Z (= toward the Sun)
      handMesh = new THREE.Mesh(new THREE.SphereGeometry(mirrorR, 28, 14, 0, Math.PI * 2, 0, Math.PI / 2.6), handMat);
      handMesh.geometry.rotateX(Math.PI / 2);
      const rim = new THREE.Mesh(new THREE.TorusGeometry(mirrorR * Math.sin(Math.PI / 2.6), .045, 10, 44), jointMat);
      rim.position.z = mirrorR * Math.cos(Math.PI / 2.6);
      handG.add(handMesh, rim);
    } else {
      handMesh = new THREE.Mesh(new THREE.SphereGeometry(.36, 18, 10, 0, Math.PI * 2, 0, Math.PI / 2.3), handMat);
      handMesh.geometry.rotateX(Math.PI / 2);
      handG.add(handMesh);
    }
    fore.add(handG);
    shoulder.add(upper, elbow, fore);
    group.add(shoulder);
    arms.push({
      shoulder, fore, handG, dir: dir.clone(), baseQ: shoulder.quaternion.clone(),
      phase: Math.random() * 7, broken: false,
      vv: new THREE.Vector3(), av: new THREE.Vector3(),
    });
  }

  // hand-tip span in the local YZ plane (the net / mirror curtain radius)
  group.updateMatrixWorld(true);
  const v = new THREE.Vector3();
  let tipSpan = 0, tipX = 0;
  for (const a of arms) { a.handG.getWorldPosition(v); group.worldToLocal(v); tipSpan += Math.hypot(v.y, v.z); tipX += v.x; }
  tipSpan /= 8; tipX /= 8;

  const api = {
    group, body, bodyMat, armMat, jointMat, handMat, arms, tipSpan, tipX,
    // idle sway of the octo-hands; broken arms tumble and drift
    update(t, dt) {
      const e = new THREE.Euler(), q = new THREE.Quaternion();
      for (const a of arms) {
        if (a.broken) {
          a.shoulder.position.addScaledVector(a.vv, dt);
          a.vv.multiplyScalar(1 - .10 * dt);
          a.shoulder.rotateX(a.av.x * dt); a.shoulder.rotateY(a.av.y * dt); a.shoulder.rotateZ(a.av.z * dt);
        } else {
          e.set(Math.sin(t * 1.2 + a.phase) * .05, Math.cos(t * .9 + a.phase) * .05, 0);
          a.shoulder.quaternion.copy(a.baseQ).multiply(q.setFromEuler(e));
          a.fore.rotation.x = Math.sin(t * 1.6 + a.phase) * .07;
        }
      }
    },
    // 0..1 overload glow on arms + hands
    setStress(h) {
      armMat.emissive.setRGB(h * .85, h * .10, h * .07);
      handMat.emissive.setRGB(h * .55, h * .08, h * .05);
      jointMat.emissive.setRGB(h * .45, h * .06, h * .04);
    },
    // detach all arms into debris; pushDir = beam/flow push direction (world)
    breakArms(parent, pushDir) {
      for (const a of arms) {
        if (a.broken) continue;
        a.broken = true;
        parent.attach(a.shoulder);                      // keep world transform
        a.vv.copy(pushDir).multiplyScalar(1.7 + Math.random() * 1.5)
          .add(new THREE.Vector3(Math.random() - .5, Math.random() - .5, Math.random() - .5).multiplyScalar(1.6));
        a.av.set((Math.random() - .5) * 4, (Math.random() - .5) * 4, (Math.random() - .5) * 4);
      }
    },
    // world positions of the 8 hands
    handTipsWorld(out) {
      for (let k = 0; k < 8; k++) arms[k].handG.getWorldPosition(out[k]);
      return out;
    },
    // local (rig-space) positions of the 8 hands
    handTipsLocal(out) {
      for (let k = 0; k < 8; k++) { arms[k].handG.getWorldPosition(v); out[k].copy(group.worldToLocal(v)); }
      return out;
    },
  };
  return api;
}

// radial-gradient sprite texture (Sun glow, flashes)
export function glowTexture(color = '255,240,180') {
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const g = c.getContext('2d');
  const gr = g.createRadialGradient(64, 64, 2, 64, 64, 64);
  gr.addColorStop(0, `rgba(${color},1)`); gr.addColorStop(.35, `rgba(${color},.45)`); gr.addColorStop(1, `rgba(${color},0)`);
  g.fillStyle = gr; g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}
