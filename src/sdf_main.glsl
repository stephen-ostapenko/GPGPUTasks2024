
float dot2(vec3 v) {
    return dot(v, v);
}

// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

// XZ plane
float sdPlane(vec3 p)
{
    return p.y;
}

// косинус который пропускает некоторые периоды, удобно чтобы махать ручкой не все время
float lazycos(float angle)
{
    int nsleep = 10;
    
    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 3) {
        return cos(angle);
    }
    
    return 1.0;
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/

float smin(float a, float b, float k) {
    k *= 1.0;
    float r = exp2(-a / k) + exp2(-b / k);
    return -k * log2(r);
}

float sdRoundCone(vec3 p, vec3 a, vec3 b, float r1, float r2) {
  // sampling independent computations (only depend on shape)
  vec3  ba = b - a;
  float l2 = dot(ba,ba);
  float rr = r1 - r2;
  float a2 = l2 - rr*rr;
  float il2 = 1.0/l2;

  // sampling dependant computations
  vec3 pa = p - a;
  float y = dot(pa,ba);
  float z = y - l2;
  float x2 = dot2( pa*l2 - ba*y );
  float y2 = y*y*l2;
  float z2 = z*z*l2;

  // single square root!
  float k = sign(rr)*rr*rr*x2;
  if( sign(z)*a2*z2>k ) return  sqrt(x2 + z2)        *il2 - r2;
  if( sign(y)*a2*y2<k ) return  sqrt(x2 + y2)        *il2 - r1;
                        return (sqrt(x2*a2*il2)+y*rr)*il2 - r1;
}

vec3 getTranspose() {
    return vec3(cos(iTime * 5.0) * 0.2, sin(iTime * 2.0) * 0.1, 0.0);
}

vec4 sdBody(vec3 p) {
    float d = 1e10;

    p -= vec3(0.0, 0.5, -0.7);
    p += getTranspose();

    // TODO
    float r1 = 0.3;
    float r2 = 0.1;
    float k = 0.2;
    float h = 0.4;

    d = smin(sdSphere(p, r1), sdSphere(p - vec3(0.0, h, 0.0), r2), k);

    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdLegs(vec3 p) {
    p -= vec3(0.0, 0.0, -0.7);

    float rLower = 0.05;
    float rUpper = 0.1;
    vec3 lowerR = vec3(-0.2, 0.0, 0.0);
    vec3 upperR = vec3(-0.1, 0.3, 0.0);
    vec3 lowerL = vec3(0.2, 0.0, 0.0);
    vec3 upperL = vec3(0.1, 0.3, 0.0);

    upperR -= getTranspose();
    upperL -= getTranspose();

    float dR = sdRoundCone(p, lowerR, upperR, rLower, rUpper);
    float dL = sdRoundCone(p, lowerL, upperL, rLower, rUpper);

    return vec4(min(dL, dR), vec3(0.0, 1.0, 0.0));
}

vec4 sdArms(vec3 p) {
    p -= vec3(0.0, 0.0, -0.7);

    float rLower = 0.02;
    float rUpper = 0.05;
    vec3 lowerR = vec3(-1.0, 0.9, 0.0);
    vec3 upperR = vec3(-0.1, 0.7, 0.0);
    vec3 lowerL = vec3(1.0, 0.9, 0.0);
    vec3 upperL = vec3(0.1, 0.7, 0.0);

    upperR -= getTranspose();
    upperL -= getTranspose();

    float dR = sdRoundCone(p, lowerR, upperR, rLower, rUpper);
    float dL = sdRoundCone(p, lowerL, upperL, rLower, rUpper);

    return vec4(min(dL, dR), vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p) {
    p -= vec3(0.0, 0.0, -0.7);

    float r = 0.2;
    vec3 center = vec3(0.0, 0.7, 0.24);
    vec3 lookAt = vec3(0.0, 0.7, 1.0);

    center -= getTranspose();

    float d = sdSphere((p - center), r);
    vec3 eyeDir = lookAt - center;
    float angle = max(0.0, dot(normalize(p - center), normalize(eyeDir)));
    vec3 color = vec3(1.0, 1.0, 1.0);

    if (angle > 0.95) {
        color = vec3(0.0, 0.0, 0.0);
    } else if (angle > 0.85) {
        color = vec3(0.0, 1.0, 1.0);
    }

    return vec4(d, color);
}

float sdCone(vec3 p, vec2 c, float h) {
  // c is the sin/cos of the angle, h is height
  // Alternatively pass q instead of (c,h),
  // which is the point at the base in 2D
  vec2 q = h*vec2(c.x/c.y,-1.0);

  vec2 w = vec2( length(p.xz), p.y );
  vec2 a = w - q*clamp( dot(w,q)/dot(q,q), 0.0, 1.0 );
  vec2 b = w - q*vec2( clamp( w.x/q.x, 0.0, 1.0 ), 1.0 );
  float k = sign( q.y );
  float d = min(dot( a, a ),dot(b, b));
  float s = max( k*(w.x*q.y-w.y*q.x),k*(w.y-q.y)  );
  return sqrt(d)*sign(s);
}

vec4 sdHat(vec3 p) {
    p -= vec3(0.0, 0.0, -0.7);

    p -= vec3(0.0, 1.3, 0.0);
    p += getTranspose();

    float r = 0.4;
    vec2 sc = vec2(0.5, 0.5);
    vec3 color = vec3(1.0, 0.0, 1.0);

    return vec4(sdCone(p, sc, r), color);
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне 
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(sin(iTime * 0.5) * 0.4, 0.08, 0.0);
    
    vec4 res = sdBody(p);
    
    vec4 eye = sdEye(p);
    if (eye.x < res.x) {
        res = eye;
    }
    
    vec4 legs = sdLegs(p);
    if (legs.x < res.x) {
        res = legs;
    }

    vec4 arms = sdArms(p);
    if (arms.x < res.x) {
        res = arms;
    }

    vec4 hat = sdHat(p);
    if (hat.x < res.x) {
        res = hat;
    }
    
    return res;
}


vec4 sdTotal(vec3 p)
{
    vec4 res = sdMonster(p);
    
    
    float dist = sdPlane(p);
    if (dist < res.x) {
        res = vec4(dist, texture(iChannel0, vec2(p.x, p.z)));
    }
    
    return res;
}

// see https://iquilezles.org/articles/normalsSDF/
vec3 calcNormal( in vec3 p ) // for function f(p)
{
    const float eps = 0.0001; // or some other value
    const vec2 h = vec2(eps,0);
    return normalize( vec3(sdTotal(p+h.xyy).x - sdTotal(p-h.xyy).x,
                           sdTotal(p+h.yxy).x - sdTotal(p-h.yxy).x,
                           sdTotal(p+h.yyx).x - sdTotal(p-h.yyx).x ) );
}


vec4 raycast(vec3 ray_origin, vec3 ray_direction)
{
    
    float EPS = 1e-3;
    
    
    // p = ray_origin + t * ray_direction;
    
    float t = 0.0;
    
    for (int iter = 0; iter < 200; ++iter) {
        vec4 res = sdTotal(ray_origin + t*ray_direction);
        t += res.x;
        if (res.x < EPS) {
            return vec4(t, res.yzw);
        }
    }

    return vec4(1e10, vec3(0.8, 0.8, 1.0));
}


float shading(vec3 p, vec3 light_source, vec3 normal)
{
    
    vec3 light_dir = normalize(light_source - p);
    
    float shading = dot(light_dir, normal);
    
    return clamp(shading, 0.5, 1.0);

}

// phong model, see https://en.wikibooks.org/wiki/GLSL_Programming/GLUT/Specular_Highlights
float specular(vec3 p, vec3 light_source, vec3 N, vec3 camera_center, float shinyness)
{
    vec3 L = normalize(p - light_source);
    vec3 R = reflect(L, N);

    vec3 V = normalize(camera_center - p);
    
    return pow(max(dot(R, V), 0.0), shinyness);
}


float castShadow(vec3 p, vec3 light_source)
{
    
    vec3 light_dir = p - light_source;
    
    float target_dist = length(light_dir);
    
    
    if (raycast(light_source, normalize(light_dir)).x + 0.001 < target_dist) {
        return 0.5;
    }
    
    return 1.0;
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.y;
    
    vec2 wh = vec2(iResolution.x / iResolution.y, 1.0);
    

    vec3 ray_origin = vec3(0.0, 0.7, 1.0);
    vec3 ray_direction = normalize(vec3(uv - 0.5*wh, -1.0));
    

    vec4 res = raycast(ray_origin, ray_direction);
    
    
    
    vec3 col = res.yzw;
    
    
    vec3 surface_point = ray_origin + res.x*ray_direction;
    vec3 normal = calcNormal(surface_point);
    
    vec3 light_source = vec3(1.0 + 2.5*sin(iTime), 10.0, 10.0);
    
    float shad = shading(surface_point, light_source, normal);
    shad = min(shad, castShadow(surface_point, light_source));
    col *= shad;
    
    float spec = specular(surface_point, light_source, normal, ray_origin, 30.0);
    col += vec3(1.0, 1.0, 1.0) * spec;
    
    
    
    // Output to screen
    fragColor = vec4(col, 1.0);
}
