// クライアントコンポーネント指定
"use client";

// React と useState フックをインポート
import React, { useRef, Suspense, useMemo, useCallback, useState } from 'react';
import { Canvas, useFrame, ThreeEvent } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

// --- シェーダーコード ---

// --- Simplex Noise GLSL (完全な実装) ---
const simplexNoise3D = `
  // Modulo 289
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }

  // Permutation polynomial: (34x^2 + x) mod 289
  vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }

  // Taylor inverse square root approximation
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

  // 3D Simplex noise function
  float snoise(vec3 v) {
    const vec2 C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 =   v - i + dot(i, C.xxx) ;

    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );

    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;

    i = mod289(i);
    vec4 p = permute( permute( permute(
               i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
             + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
             + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

    float n_ = 0.142857142857; // 1.0/7.0
    vec3  ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );

    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );

    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);

    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    m = m * m;

    return 42.0 * dot( m, vec4( dot(p0,x0), dot(p1,x1),
                                  dot(p2,x2), dot(p3,x3) ) );
  }
`;

// Vertex Shader
const vertexShader = `
  varying vec3 vNormal;
  varying vec3 vPosition;
  uniform float uTime;
  uniform vec3 uMousePos;
  uniform float uMouseInteractionTime;
  uniform float uMouseStrength;

  ${simplexNoise3D}

  float ripple(float dist, float time) {
    if (time < 0.0) return 0.0;
    float rippleSpeed = 2.0;
    float rippleWidth = 0.3;
    float decayRate = 2.5;
    float maxAmplitude = 0.3;
    float wavePhase = dist * (1.0 / rippleWidth) - time * rippleSpeed;
    float rippleShape = cos(wavePhase);
    float rippleEnvelope = smoothstep(0.0, rippleWidth * 2.0, dist) * smoothstep(rippleWidth * 4.0 + time * rippleSpeed, time * rippleSpeed, dist);
    float decay = exp(-time * decayRate);
    return rippleShape * rippleEnvelope * maxAmplitude * decay * uMouseStrength;
  }

  void main() {
    float warpFrequency = 0.6;
    float warpAmplitude = 2.0;
    float warpSpeed = 0.08;
    vec3 warpInput = position * warpFrequency + vec3(uTime * warpSpeed);
    vec3 warp = vec3(
      snoise(warpInput + vec3(0.0, 1.3, 7.1)),
      snoise(warpInput + vec3(5.2, 6.3, 2.8)),
      snoise(warpInput + vec3(9.7, 3.9, 4.5))
    ) * warpAmplitude;
    float noiseFrequency = 0.1;
    float noiseAmplitude = 0.08;
    float noiseSpeed = 0.12;
    vec3 noiseInput = (position + warp) * noiseFrequency + vec3(uTime * noiseSpeed);
    float noiseValue = snoise(noiseInput);
    float baseDisplacement = noiseValue * noiseAmplitude;
    vec4 worldVertexPos = modelMatrix * vec4(position, 1.0);
    float distToMouse = distance(worldVertexPos.xyz, uMousePos);
    float rippleDisplacement = ripple(distToMouse, uMouseInteractionTime);
    float totalDisplacement = baseDisplacement + rippleDisplacement;
    vec3 newPosition = position + normalize(normal) * totalDisplacement;
    vNormal = normalize(normalMatrix * normal);
    vec4 finalWorldPosition = modelMatrix * vec4(newPosition, 1.0);
    vPosition = finalWorldPosition.xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
  }
`;

// Fragment Shader (ノイズ適用方法を変更し、周波数を調整した版)
const fragmentShader = `
  varying vec3 vNormal;
  varying vec3 vPosition;
  uniform float uTime;

  ${simplexNoise3D}

  void main() {
    // --- 1. 元のグラデーション計算 ---
    vec3 colorStart = vec3(65.0 / 255.0, 164.0 / 255.0, 253.0 / 255.0) + vec3(0.6, 0.1, 0.6);
    vec3 colorEnd = vec3(14.0 / 255.0, 244.0 / 255.0, 255.0 / 255.0) + vec3(0.1, 0.0, 0.1);
    vec3 gradientDirection = normalize(vec3(-1.0, 1.0, 0.2));
    float dotProduct = dot(normalize(vNormal), gradientDirection);
    float baseGradientFactor = dotProduct * 0.5 + 0.5; // 基本的なグラデーション係数 [0, 1]

    // --- 2. ワールド座標に基づく3Dノイズパターン ---
    float colorNoiseFrequency = 111.5;
    float colorNoiseSpeed = 0.08;
    float noiseValue = snoise(vPosition * colorNoiseFrequency + vec3(0.0, 0.0, uTime * colorNoiseSpeed));
    float noiseOffset = noiseValue * 0.5; // [-0.5, 0.5] の範囲

    // --- 3. グラデーション係数をノイズで摂動させる ---
    float perturbationStrength = 0.4;
    float combinedFactor = clamp(baseGradientFactor + noiseOffset * perturbationStrength, 0.0, 1.0);

    // --- 4. 合成された係数で色を混合 ---
    vec3 gradientNoiseColor = mix(colorStart, colorEnd, combinedFactor);

    // --- 5. フレネル効果 ---
    vec3 viewDirection = normalize(cameraPosition - vPosition);
    float fresnelTerm = 1.0 - abs(dot(viewDirection, normalize(vNormal)));
    float fresnelFactor = pow(fresnelTerm, 3.0);
    vec3 fresnelColor = vec3(1.0, 1.0, 1.0);
    float fresnelIntensity = 0.05;
    vec3 finalColor = mix(gradientNoiseColor, fresnelColor, fresnelFactor * fresnelIntensity);

    gl_FragColor = vec4(finalColor, 1.0);
  }
`;

// --- AnimatedSphere コンポーネント ---
function AnimatedSphere() {
  // --- Ref と Uniforms の定義 ---
  const meshRef = useRef<THREE.Mesh>(null!);
  const interactionStartTime = useRef<number>(-1);
  const interactionPoint = useRef<THREE.Vector3>(new THREE.Vector3());
  const isInteracting = useRef<boolean>(false);
  const interactionStrength = useRef<number>(0);

  const uniforms = useMemo(() => ({
    uTime: { value: 0.0 },
    uMousePos: { value: new THREE.Vector3() },
    uMouseInteractionTime: { value: -1.0 },
    uMouseStrength: { value: 0.0 },
  }), []);

  // --- イベントハンドラと useFrame フックの定義 ---
  const handlePointerDown = useCallback((event: ThreeEvent<PointerEvent>) => {
    if (event.button !== 0) return;
    event.stopPropagation();
    if (event.intersections.length > 0) {
      const intersection = event.intersections[0];
      interactionPoint.current.copy(intersection.point);
      interactionStartTime.current = uniforms.uTime.value;
      isInteracting.current = true;
      interactionStrength.current = 0.5;
      uniforms.uMousePos.value.copy(interactionPoint.current);
      uniforms.uMouseInteractionTime.value = 0.0;
      uniforms.uMouseStrength.value = interactionStrength.current;
    }
  }, [uniforms]);

  const handlePointerMove = useCallback((event: ThreeEvent<PointerEvent>) => {
    if (!isInteracting.current) return;
    event.stopPropagation();
    if (event.intersections.length > 0) {
        const intersection = event.intersections[0];
        interactionPoint.current.copy(intersection.point);
        uniforms.uMousePos.value.copy(interactionPoint.current);
        uniforms.uMouseStrength.value = interactionStrength.current;
        const elapsedTime = uniforms.uTime.value - interactionStartTime.current;
        uniforms.uMouseInteractionTime.value = elapsedTime > 0 ? elapsedTime : 0;
    }
  }, [uniforms]);

  const handlePointerUp = useCallback((event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    if (isInteracting.current) {
      isInteracting.current = false;
    }
  }, [uniforms]);

  const handlePointerLeave = useCallback((event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    if (isInteracting.current) {
        isInteracting.current = false;
    }
  }, []);

  useFrame(({ clock }) => {
    const currentTime = clock.getElapsedTime();
    uniforms.uTime.value = currentTime;
    if (interactionStartTime.current >= 0 && !isInteracting.current) {
      const elapsedTime = currentTime - interactionStartTime.current;
      uniforms.uMouseInteractionTime.value = elapsedTime;
    } else if (isInteracting.current) {
        const elapsedTime = currentTime - interactionStartTime.current;
        uniforms.uMouseInteractionTime.value = elapsedTime > 0 ? elapsedTime : 0;
    }
  });

  // --- return 文 (JSX) ---
  return (
    <mesh
      ref={meshRef}
      castShadow={true}
      receiveShadow={true}
      scale={1.3}
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
      onPointerMove={handlePointerMove}
      onPointerLeave={handlePointerLeave}
    >
      <sphereGeometry args={[1, 128, 128]} />
      <shaderMaterial
        uniforms={uniforms}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
      />
    </mesh>
  );
}

// --- 立方格子コンポーネント ---
interface CubeLatticeProps {
  size?: number;
  divisions?: number;
  color?: THREE.ColorRepresentation;
  lineWidth?: number;
}

function CubeLattice({
  size = 4,
  divisions = 5,
  color = '#aaaaaa',
  lineWidth = 1,
}: CubeLatticeProps) {
    const geometry = useMemo(() => {
        const points: THREE.Vector3[] = [];
        const halfSize = size / 2;
        const step = size / divisions;
        for (let i = 0; i <= divisions; i++) {
        for (let j = 0; j <= divisions; j++) {
            points.push(new THREE.Vector3(-halfSize + i * step, -halfSize + j * step, -halfSize));
            points.push(new THREE.Vector3(-halfSize + i * step, -halfSize + j * step, halfSize));
            points.push(new THREE.Vector3(-halfSize + i * step, -halfSize, -halfSize + j * step));
            points.push(new THREE.Vector3(-halfSize + i * step, halfSize, -halfSize + j * step));
            points.push(new THREE.Vector3(-halfSize, -halfSize + i * step, -halfSize + j * step));
            points.push(new THREE.Vector3(halfSize, -halfSize + i * step, -halfSize + j * step));
        }
        }
        const bufferGeometry = new THREE.BufferGeometry().setFromPoints(points);
        return bufferGeometry;
    }, [size, divisions]);

    return (
        <lineSegments geometry={geometry}>
        <lineBasicMaterial color={color} linewidth={lineWidth} transparent opacity={0.5} />
        </lineSegments>
    );
}


// --- ページコンポーネント本体 ---
export default function HomePage() {
  // ★ 格子の表示状態を管理する State (デフォルトは表示)
  const [isLatticeVisible, setIsLatticeVisible] = useState(true);

  // ★ ボタンクリックで State をトグルする関数
  const toggleLattice = () => {
    setIsLatticeVisible(!isLatticeVisible);
  };

  return (
    <>
      {/* UI要素 (Canvas外) */}
      <div style={{ position: 'absolute', top: 0, left: 0, zIndex: 10, padding: '1rem', color: '#fff' }}>
        <h1 style={{ fontSize: '64px', fontFamily: 'Futura', margin: 0, marginBottom: '1rem' }}>
          Fluid Sphere with Shader
        </h1>
        {/* ★ 表示切替ボタン */}
        <button
          onClick={toggleLattice}
          style={{
            padding: '8px 16px',
            width: '140px',
            fontSize: '16px',
            cursor: 'pointer',
            backgroundColor: '#eee',
            color: '#999',
            border: '1px solid #ccc',
            fontFamily: 'Futura',
            position: 'absolute',
            top: '144px', left: '88px',
            transform: 'translate(-50%, -50%)',
          }}
        >
          {isLatticeVisible ? 'Hide Lattice' : 'Show Lattice'}
        </button>
      </div>

      {/* Canvas要素 */}
      <Canvas
          shadows={true}
          camera={{ position: [0, 0, 5.5], fov: 50 }}
          gl={{ antialias: true }}
          // スタイルで Canvas を背景に配置 (-1)
          style={{ width: '100%', height: '100vh', backgroundColor: '#eee', position: 'absolute', top: 0, left: 0, zIndex: -1 }}
        >
          <Suspense fallback={null}>
            <ambientLight intensity={0.5} />
            <directionalLight
              castShadow
              position={[5, 6, 4]}
              intensity={1.0}
              shadow-mapSize-width={2048}
              shadow-mapSize-height={2048}
              shadow-camera-far={20}
              shadow-camera-left={-10}
              shadow-camera-right={10}
              shadow-camera-top={10}
              shadow-camera-bottom={-10}
              shadow-bias={-0.0005}
            />
            <directionalLight position={[-3, -3, 2]} intensity={0.3} />

            <AnimatedSphere />

            {/* ★ isLatticeVisible が true の場合のみ格子をレンダリング */}
            {isLatticeVisible && <CubeLattice size={3.8} divisions={6} color="#666666" />}

            <OrbitControls enableZoom={true} enablePan={true} autoRotate={false} />
          </Suspense>
        </Canvas>
    </>
  );
}