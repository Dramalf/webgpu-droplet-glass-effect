const AIR_IOR: f32 = 1.0003;
const GLASS_IOR: f32 = 1.5;
const REFRACT_RT_DISTANCE: f32 = 250.0;
const REFLECT_RT_DISTANCE: f32 = 250.0;
const CIRCLE_SIZE: f32 = 60.0;
const EDGE_FRACTION: f32 = 0.3;
const EDGE_POWER: f32 = 4.0;
const NORMAL_JITER: f32 = 0.01;
const CHROMATIC_ABBERATION_STRENGTH: f32 = 1.0;
const BLUR_LOD_BIAS: f32 = 0.5;
const PI: f32 = 3.14159265358979323846;
const RCP_PI: f32 = 0.31830988618379067154;

// TextureNice by IQ - adapted for WebGPU
fn textureNice(tex: texture_2d<f32>, sam: sampler, uv_in: vec2<f32>, level: i32) -> vec4<f32> {
    let textureResolution = f32(textureDimensions(tex, level).x);
    var uv = uv_in * textureResolution + 0.5;
    let iuv = floor(uv);
    let fuv = fract(uv);
    uv = iuv + fuv * fuv * (3.0 - 2.0 * fuv);
    uv = (uv - 0.5) / textureResolution;
    return pow(textureSampleLevel(tex, sam, uv, f32(level)), vec4<f32>(2.2));
}

fn textureNiceTrilinear(tex: texture_2d<f32>, sam: sampler, uv: vec2<f32>, lod: f32) -> vec4<f32> {
    let interpo = fract(lod);
    let floorLod = i32(floor(lod));
    let base = textureNice(tex, sam, uv, floorLod);
    let higher = textureNice(tex, sam, uv, floorLod + 1);
    return mix(base, higher, interpo);
}

fn rand_IGN(v_in: vec2<f32>, frame_in: u32) -> f32 {
    let frame = frame_in % 64u;
    let v = v_in + 5.588238 * f32(frame);
    return fract(52.9829189 * fract(0.06711056 * v.x + 0.00583715 * v.y));
}

fn pow2(x: f32) -> f32 { return x * x; }

fn pow5(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2 * x;
}

fn rcp(x: f32) -> f32 { return 1.0 / x; }

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

fn linearStep(edge0: f32, edge1: f32, x: f32) -> f32 { 
    return saturate((x - edge0) / (edge1 - edge0)); 
}

fn fresnel_iorToF0(ior: f32) -> f32 {
    return pow2((ior - AIR_IOR) / (ior + AIR_IOR));
}

fn fresnel_schlick(cosTheta: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow5(1.0 - cosTheta);
}

fn _bsdf_g_Smith_Schlick_denom(cosTheta: f32, k: f32) -> f32 {
    return cosTheta * (1.0 - k) + k;
}

fn bsdf_ggx(roughness: f32, NDotL: f32, NDotV: f32, NDotH: f32) -> f32 {
    if (NDotL <= 0.0) { return 0.0; }
    let NDotH2 = pow2(NDotH);
    let a2 = pow2(roughness);
    let d = a2 / (PI * pow2(NDotH2 * (a2 - 1.0) + 1.0));
    let k = roughness * 0.5;
    let v = rcp(_bsdf_g_Smith_Schlick_denom(NDotL, k) * _bsdf_g_Smith_Schlick_denom(saturate(NDotV), k));
    return NDotL * d * v;
}

const INCIDENT_VECTOR: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);

fn glassIorCA(wavelength: f32) -> f32 {
    let abberation = CHROMATIC_ABBERATION_STRENGTH * 0.1;
    let glassIor = mix(GLASS_IOR + abberation, GLASS_IOR - abberation, 
                      1.0 - pow(1.0 - linearStep(450.0, 650.0, wavelength), 4.0));
    return glassIor;
}

fn sampleRefraction(tex: texture_2d<f32>, sam: sampler, fragCoord: vec2<f32>, 
                   sdfValue: f32, normal: vec3<f32>, glassIor: f32, iResolution: vec2<f32>) -> vec3<f32> {
    var refractVector = refract(INCIDENT_VECTOR, normal, AIR_IOR / glassIor);
    refractVector = refractVector / abs(refractVector.z / REFRACT_RT_DISTANCE);
    let refractedUV = (fragCoord + refractVector.xy) / iResolution;
    let refractedColor = textureNiceTrilinear(tex, sam, refractedUV, sdfValue * 2.0 + BLUR_LOD_BIAS).rgb;
    return refractedColor;
}

// 2D旋转矩阵
fn rotate2D(point: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(
        point.x * c - point.y * s,
        point.x * s + point.y * c
    );
}

// 水滴形状的SDF (Signed Distance Function)
fn dropletSDF(p: vec2<f32>, strength: f32) -> f32 {
    let circleRadius = CIRCLE_SIZE;

    // 当strength很小时，保持完美圆形（提高阈值确保静止时完全是圆形）
    if (strength < 0.1) {
        return length(p) - circleRadius;
    }
    
    // 使用平滑插值在圆形和水滴形状之间过渡
    let baseCircleDist = length(p) - circleRadius;
    
    
    let elongation = (strength - 0.1) * 0.8;  // 减去0.1避免突变
    let tapering = (strength - 0.1) * 0.6;
    
    var pos = p;
    pos.x = pos.x / (1.0 + elongation);
    
    let dist = length(pos);
    var radius = circleRadius;
    
    if (pos.x > 0.0) {
        let frontFactor = 1.0 - (pos.x / circleRadius) * tapering;
        radius = circleRadius * max(0.3, frontFactor);
    } else {
        let backFactor = 1.0 + abs(pos.x / circleRadius) * tapering * 0.2;
        radius = circleRadius * min(1.2, backFactor);
    }
    
    let dropletDist = dist - radius;
    
    // 在圆形和水滴形状之间平滑过渡
    let transitionFactor = smoothstep(0.1, 0.2, strength);
    return mix(baseCircleDist, dropletDist, transitionFactor);
}

// 动态形状SDF，结合圆形和水滴形状
fn dynamicShapeSDF(point: vec2<f32>, center: vec2<f32>, strength: f32, angle: f32) -> f32 {
    // 相对于中心的位置
    let relativePos = point - center;
    
    // 旋转到水滴的局部坐标系（尖端朝向运动方向）
    let rotatedPos = rotate2D(relativePos, -angle);
    // 计算水滴形状的SDF
    return dropletSDF(rotatedPos, strength);
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0)
    );
    
    var texCoord = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 0.0)
    );

    var output: VertexOutput;
    output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
    output.texCoord = texCoord[vertexIndex];
    return output;
}

// Fragment Shader
struct Uniforms {
    mousePos: vec2<f32>,
    time: f32,
    resolution: vec2<f32>,
    deformation: vec2<f32>,  // x: strength (0-1), y: angle (radians)
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var textureSampler: sampler;
@group(0) @binding(2) var textureData: texture_2d<f32>;

@fragment
fn fs_main(@location(0) texCoord: vec2<f32>) -> @location(0) vec4<f32> {
    let iResolution = uniforms.resolution;
    let fragCoord = texCoord * iResolution;
    let iFrame = u32(uniforms.time * 60.0); // Approximate frame count
    
    // Background texture
    let bg = textureNice(textureData, textureSampler, texCoord, 0).rgb;
    var color = bg;
    
    let randV = 0.0;//rand_IGN(fragCoord, iFrame);
    let randAngle = randV * PI * 2.0;
    
    // Mouse position - convert from UV to screen coordinates
    let mouseScreenPos = uniforms.mousePos * iResolution;
    let circleCenter = select(mouseScreenPos, iResolution * 0.5, 
                             all(uniforms.mousePos == vec2<f32>(0.0)));
    
    // 提取形变参数
    let deformationStrength = uniforms.deformation.x;
    let deformationAngle = uniforms.deformation.y;
    
    // 使用动态形状SDF计算距离
    let shapeDist = dynamicShapeSDF(fragCoord, circleCenter, deformationStrength, deformationAngle);
    let sdfValue = pow(linearStep(CIRCLE_SIZE * EDGE_FRACTION, CIRCLE_SIZE, shapeDist), EDGE_POWER);
    
    // 计算水滴形状的法线
    let epsilon = 0.01;
    let gradX = dynamicShapeSDF(fragCoord + vec2<f32>(epsilon, 0.0), circleCenter, deformationStrength, deformationAngle) - 
               dynamicShapeSDF(fragCoord - vec2<f32>(epsilon, 0.0), circleCenter, deformationStrength, deformationAngle);
    let gradY = dynamicShapeSDF(fragCoord + vec2<f32>(0.0, epsilon), circleCenter, deformationStrength, deformationAngle) - 
               dynamicShapeSDF(fragCoord - vec2<f32>(0.0, epsilon), circleCenter, deformationStrength, deformationAngle);
    
    let surfaceNormal = normalize(vec2<f32>(gradX, gradY));
    
    // 混合计算的法线和一些随机扰动
    var normal = mix(
        normalize(vec3<f32>(sin(randAngle), cos(randAngle), -rcp(NORMAL_JITER))), 
        vec3<f32>(surfaceNormal, 0.0), 
        sdfValue
    );
    normal = normalize(normal);
    
    // Chromatic aberration refraction
    var refractedColor = vec3<f32>(0.0);
    if (CHROMATIC_ABBERATION_STRENGTH > 0.0) {
        refractedColor += sampleRefraction(textureData, textureSampler, fragCoord, sdfValue, normal, glassIorCA(611.4), iResolution) * vec3<f32>(1.0, 0.0, 0.0);
        refractedColor += sampleRefraction(textureData, textureSampler, fragCoord, sdfValue, normal, glassIorCA(570.5), iResolution) * vec3<f32>(1.0, 1.0, 0.0);
        refractedColor += sampleRefraction(textureData, textureSampler, fragCoord, sdfValue, normal, glassIorCA(549.1), iResolution) * vec3<f32>(0.0, 1.0, 0.0);
        refractedColor += sampleRefraction(textureData, textureSampler, fragCoord, sdfValue, normal, glassIorCA(491.4), iResolution) * vec3<f32>(0.0, 1.0, 1.0);
        refractedColor += sampleRefraction(textureData, textureSampler, fragCoord, sdfValue, normal, glassIorCA(464.2), iResolution) * vec3<f32>(0.0, 0.0, 1.0);
        refractedColor += sampleRefraction(textureData, textureSampler, fragCoord, sdfValue, normal, glassIorCA(374.0), iResolution) * vec3<f32>(1.0, 0.0, 1.0);
        refractedColor = refractedColor / 3.0;
    } else {
        refractedColor = sampleRefraction(textureData, textureSampler, fragCoord, sdfValue, normal, GLASS_IOR, iResolution);
    }
    
    let V = vec3<f32>(0.0, 0.0, -1.0);
    let NDotV = saturate(dot(V, normal));
    
    let fresnelV = fresnel_schlick(NDotV, fresnel_iorToF0(GLASS_IOR));
    
    var reflectVector = reflect(INCIDENT_VECTOR, normal);
    let L = reflectVector;
    let H = normalize(L + V);
    reflectVector = reflectVector / abs(reflectVector.z / REFLECT_RT_DISTANCE);
    
    let reflectedUV = (fragCoord + reflectVector.xy) / iResolution;
    var reflectedColor = textureNiceTrilinear(textureData, textureSampler, reflectedUV, 2.5 + BLUR_LOD_BIAS).rgb;
    
    let NDotL = dot(normal, L);
    let NDotH = dot(normal, H);
    
    let ggx = bsdf_ggx(0.5, NDotL, NDotV, NDotH);
    reflectedColor = reflectedColor * ggx;
    
    let glassColor = mix(refractedColor, reflectedColor, fresnelV);
    
    // 使用形状距离进行混合，创建平滑边缘
    let edgeBlend = smoothstep(CIRCLE_SIZE, CIRCLE_SIZE - 2.0, abs(shapeDist));
    color = mix(color, glassColor, edgeBlend);
    
    // Gamma correction
    color = pow(color, vec3<f32>(1.0 / 2.2));
    
    return vec4<f32>(color, 1.0);
} 