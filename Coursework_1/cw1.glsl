#define SOLUTION_CYLINDER_AND_PLANE
#define SOLUTION_SHADOW
//#define SOLUTION_REFLECTION_REFRACTION
//#define SOLUTION_FRESNEL
//#define SOLUTION_BOOLEAN

precision highp float;
uniform ivec2 viewport; 

struct PointLight {
    vec3 position;
    vec3 color;
};

struct Material {
    vec3  diffuse;
    vec3  specular;
    float glossiness;
    float reflection;
    float refraction;
    float ior;
};

struct Sphere {
    vec3 position;
    float radius;
    Material material;
};

struct Plane {
    vec3 normal;
    float d;
    Material material;
};

struct Cylinder {
    vec3 position;
    vec3 direction;
    float radius;
    Material material;
};

#define BOOLEAN_MODE_AND 0			// and 
#define BOOLEAN_MODE_MINUS 1			// minus 

struct Boolean {
    Sphere spheres[2];
    int mode;
};


const int lightCount = 2;
const int sphereCount = 3;
const int planeCount = 1;
const int cylinderCount = 2;
const int booleanCount = 2; 

struct Scene {
    vec3 ambient;
    PointLight[lightCount] lights;
    Sphere[sphereCount] spheres;
    Plane[planeCount] planes;
    Cylinder[cylinderCount] cylinders;
    Boolean[booleanCount] booleans;
};

struct Ray {
    vec3 origin;
    vec3 direction;
};

// Contains all information pertaining to a ray/object intersection
struct HitInfo {
    bool hit;
    float t;
    vec3 position;
    vec3 normal;
    Material material;
    bool enteringPrimitive;
};

HitInfo getEmptyHit() {
    return HitInfo(
                   false,
                   0.0,
                   vec3(0.0),
                   vec3(0.0),
                   Material(vec3(0.0), vec3(0.0), 0.0, 0.0, 0.0, 0.0),
                   false);
}

// Sorts the two t values such that t1 is smaller than t2
void sortT(inout float t1, inout float t2) {
    // Make t1 the smaller t
    if(t2 < t1)  {
        float temp = t1;
        t1 = t2;
        t2 = temp;
    }
}

// Tests if t is in an interval
bool isTInInterval(const float t, const float tMin, const float tMax) {
    return t > tMin && t < tMax;
}

// Get the smallest t in an interval.
bool getSmallestTInInterval(float t0, float t1, const float tMin, const float tMax, inout float smallestTInInterval) {
    
    sortT(t0, t1);
    
    // As t0 is smaller, test this first
    if(isTInInterval(t0, tMin, tMax)) {
        smallestTInInterval = t0;
        return true;
    }
    
    // If t0 was not in the interval, still t1 could be
    if(isTInInterval(t1, tMin, tMax)) {
        smallestTInInterval = t1;
        return true;
    }
    
    // None was
    return false;
}

HitInfo intersectSphere(const Ray ray, const Sphere sphere, const float tMin, const float tMax) {
    
    vec3 to_sphere = ray.origin - sphere.position;
    
    float a = dot(ray.direction, ray.direction);
    float b = 2.0 * dot(ray.direction, to_sphere);
    float c = dot(to_sphere, to_sphere) - sphere.radius * sphere.radius;
    float D = b * b - 4.0 * a * c;
    if (D > 0.0)
    {
        float t0 = (-b - sqrt(D)) / (2.0 * a);
        float t1 = (-b + sqrt(D)) / (2.0 * a);
        
        float smallestTInInterval;
        if(!getSmallestTInInterval(t0, t1, tMin, tMax, smallestTInInterval)) {
            return getEmptyHit();
        }
        
        vec3 hitPosition = ray.origin + smallestTInInterval * ray.direction;
        
        //Checking if we're inside the sphere by checking if the ray's origin is inside. If we are, then the normal
        //at the intersection surface points towards the center. Otherwise, if we are outside the sphere, then the normal
        //at the intersection surface points outwards from the sphere's center. This is important for refraction.
        vec3 normal =
        length(ray.origin - sphere.position) < sphere.radius + 0.001?
        -normalize(hitPosition - sphere.position):
        normalize(hitPosition - sphere.position);
        
        //Checking if we're inside the sphere by checking if the ray's origin is inside,
        // but this time for IOR bookkeeping.
        //If we are inside, set a flag to say we're leaving. If we are outside, set the flag to say we're entering.
        //This is also important for refraction.
        bool enteringPrimitive =
        length(ray.origin - sphere.position) < sphere.radius + 0.001 ?
        false:
        true;
        
        return HitInfo(
                       true,
                       smallestTInInterval,
                       hitPosition,
                       normal,
                       sphere.material,
                       enteringPrimitive);
    }
    return getEmptyHit();
}

HitInfo intersectPlane(const Ray ray, const Plane plane, const float tMin, const float tMax) {
#ifdef SOLUTION_CYLINDER_AND_PLANE
    //t = (-d - normal dot origin) / (normal dot direction)
    float t = (plane.d - dot(ray.origin, plane.normal)) / dot(plane.normal, ray.direction);
    if (t >= 0.) {
        if (!isTInInterval(t, tMin, tMax)) {
            return getEmptyHit();
        }
        vec3 hitPosition = ray.origin + t * ray.direction;
        return HitInfo(
                       true,
                       t,
                       hitPosition,
                       plane.normal,
                       plane.material,
                       (dot(plane.normal, ray.origin) + plane.d) > 0. ? true : false
                       );
    }
#endif  
    return getEmptyHit();
}

float lengthSquared(vec3 x) {
    return dot(x, x);
}

HitInfo intersectCylinder(const Ray ray, const Cylinder cylinder, const float tMin, const float tMax) {
#ifdef SOLUTION_CYLINDER_AND_PLANE
    float cx = cylinder.direction.x, cy = cylinder.direction.y, cz = cylinder.direction.z;
    float rx = ray.direction.x, ry = ray.direction.y, rz = ray.direction.z;
    float crx = cylinder.position.x - ray.origin.x, cry = cylinder.position.y - ray.origin.y, crz = cylinder.position.z - ray.origin.z;
    float AX = cz * ry - cy * rz, AY = cx * rz - cz * rx, AZ = cy * rx - cx * ry;
    float BX = cy * crz - cz * cry, BY = cz * crx - cx * crz, BZ = cx * cry - cy * crx;
    float a = AX * AX + AY * AY + AZ * AZ;
    float b = 2. * (AX * BX + AY * BY + AZ * BZ);
    float c = BX * BX + BY * BY + BZ * BZ - lengthSquared(cylinder.direction) * cylinder.radius * cylinder.radius;
    float D = b * b - 4.0 * a * c;
    if (D > 0.0)
    {
        float t0 = (-b - sqrt(D)) / (2.0 * a);
        float t1 = (-b + sqrt(D)) / (2.0 * a);
        
        float smallestTInInterval;
        if(!getSmallestTInInterval(t0, t1, tMin, tMax, smallestTInInterval)) {
            return getEmptyHit();
        }
        
        vec3 hitPosition = ray.origin + smallestTInInterval * ray.direction;
        vec3 footPoint = cylinder.position - (dot(cylinder.direction, cylinder.position - hitPosition) / lengthSquared(cylinder.direction)) * cylinder.direction;
        float originToCylinder = length(cross(cylinder.direction, cylinder.position - ray.origin)) / length(cylinder.direction);
        
        //Checking if we're inside the sphere by checking if the ray's origin is inside. If we are, then the normal
        //at the intersection surface points towards the center. Otherwise, if we are outside the sphere, then the normal
        //at the intersection surface points outwards from the sphere's center. This is important for refraction.
        vec3 normal =
        originToCylinder < cylinder.radius + 0.001?
        -normalize(hitPosition - footPoint):
        normalize(hitPosition - footPoint);
        
        //Checking if we're inside the sphere by checking if the ray's origin is inside,
        // but this time for IOR bookkeeping.
        //If we are inside, set a flag to say we're leaving. If we are outside, set the flag to say we're entering.
        //This is also important for refraction.
        bool enteringPrimitive =
        length(hitPosition - footPoint) < cylinder.radius + 0.001 ?
        false:
        true;
        
        return HitInfo(
                       true,
                       smallestTInInterval,
                       hitPosition,
                       normal,
                       cylinder.material,
                       enteringPrimitive);
    }
#endif  
    return getEmptyHit();
}

bool inside(const vec3 position, const Sphere sphere) {
    return length(position - sphere.position) < sphere.radius;
}

HitInfo intersectBoolean(const Ray ray, const Boolean boolean, const float tMin, const float tMax) {
#ifdef SOLUTION_BOOLEAN
#else
    // Put your code for the boolean task in the #ifdef above!
#endif
    return getEmptyHit();
}

uniform float time;

HitInfo getBetterHitInfo(const HitInfo oldHitInfo, const HitInfo newHitInfo) {
    if(newHitInfo.hit)
        if(newHitInfo.t < oldHitInfo.t)  // No need to test for the interval, this has to be done per-primitive
            return newHitInfo;
    return oldHitInfo;
}

HitInfo intersectScene(const Scene scene, const Ray ray, const float tMin, const float tMax) {
    HitInfo bestHitInfo;
    bestHitInfo.t = tMax;
    bestHitInfo.hit = false;
    
    
    for (int i = 0; i < booleanCount; ++i) {
        bestHitInfo = getBetterHitInfo(bestHitInfo, intersectBoolean(ray, scene.booleans[i], tMin, tMax));
    }
    
    for (int i = 0; i < planeCount; ++i) {
        bestHitInfo = getBetterHitInfo(bestHitInfo, intersectPlane(ray, scene.planes[i], tMin, tMax));
    }
    
    for (int i = 0; i < sphereCount; ++i) {
        bestHitInfo = getBetterHitInfo(bestHitInfo, intersectSphere(ray, scene.spheres[i], tMin, tMax));
    }
    
    for (int i = 0; i < cylinderCount; ++i) {
        bestHitInfo = getBetterHitInfo(bestHitInfo, intersectCylinder(ray, scene.cylinders[i], tMin, tMax));
    }
    
    return bestHitInfo;
}

vec3 shadeFromLight(
                    const Scene scene,
                    const Ray ray,
                    const HitInfo hit_info,
                    const PointLight light)
{ 
    vec3 hitToLight = light.position - hit_info.position;
    
    vec3 lightDirection = normalize(hitToLight);
    vec3 viewDirection = normalize(hit_info.position - ray.origin);
    vec3 reflectedDirection = reflect(viewDirection, hit_info.normal);
    float diffuse_term = max(0.0, dot(lightDirection, hit_info.normal));
    float specular_term  = pow(max(0.0, dot(lightDirection, reflectedDirection)), hit_info.material.glossiness);
    
#ifdef SOLUTION_SHADOW
    float visibility = 1.0;
#else
    // Put your shadow test here
    float visibility = 1.0;
#endif
    return 	visibility *
    light.color * (
                   specular_term * hit_info.material.specular +
                   diffuse_term * hit_info.material.diffuse);
}

vec3 background(const Ray ray) {
    // A simple implicit sky that can be used for the background
    return vec3(0.2) + vec3(0.8, 0.6, 0.5) * max(0.0, ray.direction.y);
}

// It seems to be a WebGL issue that the third parameter needs to be inout instea dof const on Tobias' machine
vec3 shade(const Scene scene, const Ray ray, inout HitInfo hitInfo) {
    
    if(!hitInfo.hit) {
        return background(ray);
    }
    
    vec3 shading = scene.ambient * hitInfo.material.diffuse;
    for (int i = 0; i < lightCount; ++i) {
        shading += shadeFromLight(scene, ray, hitInfo, scene.lights[i]);
    }
    return shading;
}


Ray getFragCoordRay(const vec2 frag_coord) {
    float sensorDistance = 1.0;
    vec2 sensorMin = vec2(-1, -0.5);
    vec2 sensorMax = vec2(1, 0.5);
    vec2 pixelSize = (sensorMax- sensorMin) / vec2(viewport.x, viewport.y);
    vec3 origin = vec3(0, 0, sensorDistance);
    vec3 direction = normalize(vec3(sensorMin + pixelSize * frag_coord, -sensorDistance));
    
    return Ray(origin, direction);
}

float fresnel(const vec3 viewDirection, const vec3 normal) {
#ifdef SOLUTION_FRESNEL
#else
    // Put your code to compute the Fresnel effect in the ifdef above
    return 1.0;
#endif
}

vec3 colorForFragment(const Scene scene, const vec2 fragCoord) {
    
    Ray initialRay = getFragCoordRay(fragCoord);
    HitInfo initialHitInfo = intersectScene(scene, initialRay, 0.001, 10000.0);
    vec3 result = shade(scene, initialRay, initialHitInfo);
    
    Ray currentRay;
    HitInfo currentHitInfo;
    
    // Compute the reflection
    currentRay = initialRay;
    currentHitInfo = initialHitInfo;
    
    // The initial strength of the reflection
    float reflectionWeight = 1.0;
    
    const int maxReflectionStepCount = 2;
    for(int i = 0; i < maxReflectionStepCount; i++) {
        
        if(!currentHitInfo.hit) break;
        
#ifdef SOLUTION_REFLECTION_REFRACTION
#else
        // Put your reflection weighting code in the ifdef above
#endif
        
#ifdef SOLUTION_FRESNEL
#else
        // Replace with Fresnel code in the ifdef above
        reflectionWeight *= 0.5;
#endif
        
        Ray nextRay;
#ifdef SOLUTION_REFLECTION_REFRACTION
#else
        // Put your code to compute the reflection ray in the ifdef above
#endif
        currentRay = nextRay;
        
        currentHitInfo = intersectScene(scene, currentRay, 0.001, 10000.0);
        
        result += reflectionWeight * shade(scene, currentRay, currentHitInfo);
    }
    
    // Compute the refraction
    currentRay = initialRay;
    currentHitInfo = initialHitInfo;
    
    // The initial medium is air
    float currentIOR = 1.0;
    
    // The initial strength of the refraction.
    float refractionWeight = 1.0;
    
    const int maxRefractionStepCount = 2;
    for(int i = 0; i < maxRefractionStepCount; i++) {
        
#ifdef SOLUTION_REFLECTION_REFRACTION
#else
        // Put your refraction weighting code in the ifdef above
        reflectionWeight *= 0.5;
#endif
        
#ifdef SOLUTION_FRESNEL
#else
        // Put your Fresnel code in the ifdef above
#endif      
        
        Ray nextRay;
        
        
#ifdef SOLUTION_REFLECTION_REFRACTION      
#else
        float sourceIOR;
        float destIOR;
        // Put your code to compute the reflection ray and track the IOR in the ifdef above
#endif
        currentHitInfo = intersectScene(scene, currentRay, 0.001, 10000.0);
        
        result += refractionWeight * shade(scene, currentRay, currentHitInfo);
        
        if(!currentHitInfo.hit) break;
    }
    return result;
}

Material getDefaultMaterial() {
    return Material(vec3(0.3), vec3(0), 0.0, 0.0, 0.0, 0.0);
}

Material getPaperMaterial() {
    return Material(vec3(0.7, 0.7, 0.7), vec3(0, 0, 0), 5.0, 0.0, 0.0, 0.0);
}

Material getPlasticMaterial() {
    return Material(vec3(0.9, 0.3, 0.1), vec3(1.0), 10.0, 0.9, 0.0, 0.0);
}

Material getGlassMaterial() {
    return Material(vec3(0.0), vec3(0.0), 5.0, 1.0, 1.0, 1.5);
}

Material getSteelMirrorMaterial() {
    return Material(vec3(0.1), vec3(0.3), 20.0, 0.8, 0.0, 0.0);
}

Material getMetaMaterial() {
    return Material(vec3(0.1, 0.2, 0.5), vec3(0.3, 0.7, 0.9), 20.0, 0.8, 0.0, 0.0);
}

vec3 tonemap(const vec3 radiance) {
    const float monitorGamma = 2.0;
    return pow(radiance, vec3(1.0 / monitorGamma));
}

void main() {
    // Setup scene
    Scene scene;
    scene.ambient = vec3(0.12, 0.15, 0.2);
    
    scene.lights[0].position = vec3(5, 15, -5);
    scene.lights[0].color    = 0.5 * vec3(0.9, 0.5, 0.1);
    
    scene.lights[1].position = vec3(-15, 5, 2);
    scene.lights[1].color    = 0.5 * vec3(0.1, 0.3, 1.0);
    
    // Primitives
    bool soloBoolean = false;
    
#ifdef SOLUTION_BOOLEAN
#endif
    
    if(!soloBoolean) {
        scene.spheres[0].position            	= vec3(10, -5, -16);
        scene.spheres[0].radius              	= 6.0;
        scene.spheres[0].material 				= getPaperMaterial();
        
        scene.spheres[1].position            	= vec3(-7, -2, -13);
        scene.spheres[1].radius             	= 4.0;
        scene.spheres[1].material				= getPlasticMaterial();
        
        scene.spheres[2].position            	= vec3(0, 0.5, -5);
        scene.spheres[2].radius              	= 2.0;
        scene.spheres[2].material   			= getGlassMaterial();
        
        scene.planes[0].normal            		= normalize(vec3(0, 0.8, 0));
        scene.planes[0].d              			= -4.5;
        scene.planes[0].material				= getSteelMirrorMaterial();
        
        scene.cylinders[0].position            	= vec3(-1, 1, -26);
        scene.cylinders[0].direction            = normalize(vec3(-2, 2, -1));
        scene.cylinders[0].radius         		= 1.5;
        scene.cylinders[0].material				= getPaperMaterial();
        
        scene.cylinders[1].position            	= vec3(4, 1, -5);
        scene.cylinders[1].direction            = normalize(vec3(1, 4, 1));
        scene.cylinders[1].radius         		= 0.4;
        scene.cylinders[1].material				= getPlasticMaterial();
        
    } else {
        scene.booleans[0].mode = BOOLEAN_MODE_MINUS;
        
        // sphere A
        scene.booleans[0].spheres[0].position      	= vec3(3, 0, -10);
        scene.booleans[0].spheres[0].radius      	= 2.75;
        scene.booleans[0].spheres[0].material      	= getPaperMaterial();
        
        // sphere B
        scene.booleans[0].spheres[1].position      	= vec3(6, 1, -13);
        scene.booleans[0].spheres[1].radius      	= 4.0;
        scene.booleans[0].spheres[1].material      	= getPaperMaterial();
        
        
        scene.booleans[1].mode = BOOLEAN_MODE_AND;
        
        scene.booleans[1].spheres[0].position      	= vec3(-3.0, 1, -12);
        scene.booleans[1].spheres[0].radius      	= 4.0;
        scene.booleans[1].spheres[0].material      	= getPaperMaterial();
        
        scene.booleans[1].spheres[1].position      	= vec3(-6.0, 1, -12);
        scene.booleans[1].spheres[1].radius      	= 4.0;
        scene.booleans[1].spheres[1].material      	= getMetaMaterial();
        
        
        scene.planes[0].normal            		= normalize(vec3(0, 0.8, 0));
        scene.planes[0].d              			= -4.5;
        scene.planes[0].material				= getSteelMirrorMaterial();
        
        scene.lights[0].position = vec3(-5, 25, -5);
        scene.lights[0].color    = vec3(0.9, 0.5, 0.1);
        
        scene.lights[1].position = vec3(-15, 5, 2);
        scene.lights[1].color    = 0.0 * 0.5 * vec3(0.1, 0.3, 1.0);
        
    }
    
    // Compute color for fragment
    gl_FragColor.rgb = tonemap(colorForFragment(scene, gl_FragCoord.xy));
    gl_FragColor.a = 1.0;
    
}
