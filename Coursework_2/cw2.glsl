 
#define SOLUTION_RASTERIZATION
//#define SOLUTION_CLIPPING
//#define SOLUTION_INTERPOLATION
//#define SOLUTION_ZBUFFERING
//#define SOLUTION_AALIAS
//#define SOLUTION_TEXTURING

precision highp float;
uniform float time;

// Polygon / vertex functionality
const int MAX_VERTEX_COUNT = 8;

uniform ivec2 viewport;

struct Vertex {
    vec4 position;
    vec3 color;
	vec2 texCoord;
};

const int TEXTURE_NONE = 0;
const int TEXTURE_CHECKERBOARD = 1;
const int TEXTURE_POLKADOT = 2;
const int TEXTURE_VORONOI = 3;

const int globalPrngSeed = 7;

struct Polygon {
    // Numbers of vertices, i.e., points in the polygon
    int vertexCount;
    // The vertices themselves
    Vertex vertices[MAX_VERTEX_COUNT];
	int textureType;
};

// Appends a vertex to a polygon
void appendVertexToPolygon(inout Polygon polygon, Vertex element) {
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i == polygon.vertexCount) {
            polygon.vertices[i] = element;
        }
    }
    polygon.vertexCount++;
}

// Copy Polygon source to Polygon destination
void copyPolygon(inout Polygon destination, Polygon source) {
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        destination.vertices[i] = source.vertices[i];
    }
    destination.vertexCount = source.vertexCount;
	destination.textureType = source.textureType;
}

// Get the i-th vertex from a polygon, but when asking for the one behind the last, get the first again
Vertex getWrappedPolygonVertex(Polygon polygon, int index) {
    if (index >= polygon.vertexCount) index -= polygon.vertexCount;
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i == index) return polygon.vertices[i];
    }
}

// Creates an empty polygon
void makeEmptyPolygon(out Polygon polygon) {
  polygon.vertexCount = 0;
}

// Clipping part

#define ENTERING 0
#define LEAVING 1
#define OUTSIDE 2
#define INSIDE 3

int getCrossType(Vertex poli1, Vertex poli2, Vertex wind1, Vertex wind2) {
#ifdef SOLUTION_CLIPPING
    // TODO
#else
    return INSIDE;
#endif
}

// This function assumes that the segments are not parallel or collinear.
Vertex intersect2D(Vertex a, Vertex b, Vertex c, Vertex d) {
#ifdef SOLUTION_CLIPPING
    // TODO
#else
    return a;
#endif
}

void sutherlandHodgmanClip(Polygon unclipped, Polygon clipWindow, out Polygon result) {
    Polygon clipped;
    copyPolygon(clipped, unclipped);

    // Loop over the clip window
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i >= clipWindow.vertexCount) break;

        // Make a temporary copy of the current clipped polygon
        Polygon oldClipped;
        copyPolygon(oldClipped, clipped);

        // Set the clipped polygon to be empty
        makeEmptyPolygon(clipped);

        // Loop over the current clipped polygon
        for (int j = 0; j < MAX_VERTEX_COUNT; ++j) {
            if (j >= oldClipped.vertexCount) break;
            
            // Handle the j-th vertex of the clipped polygon. This should make use of the function 
            // intersect() to be implemented above.
#ifdef SOLUTION_CLIPPING
            // TODO
#else
            appendVertexToPolygon(clipped, getWrappedPolygonVertex(oldClipped, j));
#endif
        }
    }

    // Copy the last version to the output
    copyPolygon(result, clipped);
	clipped.textureType = unclipped.textureType;
}

// SOLUTION_RASTERIZATION and culling part

#define INNER_SIDE 0
#define OUTER_SIDE 1

// Assuming a clockwise (vertex-wise) polygon, returns whether the input point 
// is on the inner or outer side of the edge (ab)
int edge(vec2 point, Vertex a, Vertex b) {
#ifdef SOLUTION_RASTERIZATION
    // TODO
    vec2 ab, ap;
    ab.x = b.position.x - a.position.x;
    ab.y = b.position.y - a.position.y;
    ap.x = point.x - a.position.x;
    ap.y = point.y - a.position.y;
    if (ab.x * ap.y - ap.x * ab.y < 0.) {
        return INNER_SIDE;
    } else {
        return OUTER_SIDE;
    }

#endif
    return OUTER_SIDE;
}

// Returns if a point is inside a polygon or not
bool isPointInPolygon(vec2 point, Polygon polygon) {
    // Don't evaluate empty polygons
    if (polygon.vertexCount == 0) return false;
    // Check against each edge of the polygon
    bool rasterise = true;
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i < polygon.vertexCount) {
#ifdef SOLUTION_RASTERIZATION
            if (edge(point, getWrappedPolygonVertex(polygon, i), getWrappedPolygonVertex(polygon, i + 1)) == OUTER_SIDE) {
                return false;
            }
#else
            rasterise = false;
#endif
        }
    }
    return rasterise;
}

bool isPointOnPolygonVertex(vec2 point, Polygon polygon) {
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i < polygon.vertexCount) {
          	ivec2 pixelDifference = ivec2(abs(polygon.vertices[i].position.xy - point) * vec2(viewport));
          	int pointSize = viewport.x / 200;
            if( pixelDifference.x <= pointSize && pixelDifference.y <= pointSize) {
              return true;
            }
        }
    }
    return false;
}

float triangleArea(vec2 a, vec2 b, vec2 c) {
    // https://en.wikipedia.org/wiki/Heron%27s_formula
    float ab = length(a - b);
    float bc = length(b - c);
    float ca = length(c - a);
    float s = (ab + bc + ca) / 2.0;
    return sqrt(max(0.0, s * (s - ab) * (s - bc) * (s - ca)));
}

Vertex interpolateVertex(vec2 point, Polygon polygon) {
    vec3 colorSum = vec3(0.0);
    vec4 positionSum = vec4(0.0);
	vec2 texCoordSum = vec2(0.0);
    float weight_sum = 0.0;
	float weight_corr_sum = 0.0;
    
	for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i < polygon.vertexCount) {
#if defined(SOLUTION_INTERPOLATION) || defined(SOLUTION_ZBUFFERING)
            // TODO
#endif

#ifdef SOLUTION_ZBUFFERING
            // TODO
#endif

#ifdef SOLUTION_INTERPOLATION
            // TODO
#endif

#ifdef SOLUTION_TEXTURING
#endif
        }
    }
    Vertex result = polygon.vertices[0];
  
#ifdef SOLUTION_INTERPOLATION
    // TODO
#endif
#ifdef SOLUTION_ZBUFFERING
    // TODO
#endif

#ifdef SOLUTION_TEXTURING
#endif 

  return result;
}

// Projection part

// Used to generate a projection matrix.
mat4 computeProjectionMatrix() {
    mat4 projectionMatrix = mat4(1);
  
  	float aspect = float(viewport.x) / float(viewport.y);  
  	float imageDistance = 2.0;
		
	float xMin = -0.5;
	float yMin = -0.5;
	float xMax = 0.5;
	float yMax = 0.5;

	
    mat4 regPyr = mat4(1.0);
    float d = imageDistance; 
		
    float w = xMax - xMin;
    float h = (yMax - yMin) / aspect;
    float x = xMax + xMin; 
    float y = yMax + yMin; 
	
    regPyr[0] = vec4(d / w, 0, 0, 0);
    regPyr[1] = vec4(0, d / h, 0, 0);
	regPyr[2] = vec4(-x/w, -y/h, 1, 0);
	regPyr[3] = vec4(0,0,0,1);
	
    // Scale by 1/D
    mat4 scaleByD = mat4(1.0/d);
    scaleByD[3][3] = 1.0;

	// Perspective Division
	mat4 perspDiv = mat4(1.0);
	perspDiv[2][3] = 1.0;
	
    projectionMatrix = perspDiv * scaleByD * regPyr;
	
  
    return projectionMatrix;
}

// Used to generate a simple "look-at" camera. 
mat4 computeViewMatrix(vec3 VRP, vec3 TP, vec3 VUV) {
    mat4 viewMatrix = mat4(1);

	// The VPN is pointing away from the TP. Can also be modeled the other way around.
    vec3 VPN = TP - VRP;
  
    // Generate the camera axes.
    vec3 n = normalize(VPN);
    vec3 u = normalize(cross(VUV, n));
    vec3 v = normalize(cross(n, u));

    viewMatrix[0] = vec4(u[0], v[0], n[0], 0);
    viewMatrix[1] = vec4(u[1], v[1], n[1], 0);
    viewMatrix[2] = vec4(u[2], v[2], n[2], 0);
    viewMatrix[3] = vec4(-dot(VRP, u), -dot(VRP, v), -dot(VRP, n), 1);
    return viewMatrix;
}

vec3 getCameraPosition() {  
    //return 10.0 * vec3(sin(time * 1.3), 0, cos(time * 1.3));
	return 10.0 * vec3(sin(0.0), 0, cos(0.0));
}

// Takes a single input vertex and projects it using the input view and projection matrices
vec4 projectVertexPosition(vec4 position) {

  // Set the parameters for the look-at camera.
    vec3 TP = vec3(0, 0, 0);
  	vec3 VRP = getCameraPosition();
    vec3 VUV = vec3(0, 1, 0);
  
    // Compute the view matrix.
    mat4 viewMatrix = computeViewMatrix(VRP, TP, VUV);

  // Compute the projection matrix.
    mat4 projectionMatrix = computeProjectionMatrix();
  
    vec4 projectedVertex = projectionMatrix * viewMatrix * position;
    projectedVertex.xyz = (projectedVertex.xyz / projectedVertex.w);
    return projectedVertex;
}

// Projects all the vertices of a polygon
void projectPolygon(inout Polygon projectedPolygon, Polygon polygon) {
    copyPolygon(projectedPolygon, polygon);
    for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
        if (i < polygon.vertexCount) {
            projectedPolygon.vertices[i].position = projectVertexPosition(polygon.vertices[i].position);
        }
    }
}

int intModulo(int a, int b)
{
	// Manual implementation of mod for int; note the % operator & mod for int isn't supported in some WebGL versions.
	return a - (a/b)*b;
}


vec3 textureCheckerboard(vec2 texCoord)
{
	#ifdef SOLUTION_TEXTURING
	#endif
	return vec3(1.0, 0.0, 0.0); 
}

int prngSeed = 5;
const int prngMult = 174763; // This is a prime
const float maxUint = 2147483647.0; // Max magnitude of a 32-bit signed integer

float prngUniform01()
{
	// Very basic linear congruential generator (https://en.wikipedia.org/wiki/Lehmer_random_number_generator)
	// Using signed integers (as some WebGL doesn't support unsigned).
	prngSeed *= prngMult;
	// Now the seed is a "random" value between -2147483648 and 2147483647. 
	// Convert to float and scale to the 0,1 range.
	float val = float(prngSeed) / maxUint;
	return 0.5 + (val * 0.5);
}

float prngUniform(float min, float max)
{
	return prngUniform01() * (max - min) + min;
}

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec3 randomColor()
{
	return hsv2rgb(vec3(prngUniform01(), prngUniform(0.4, 1.0), prngUniform(0.7, 1.0)));
}

vec3 texturePolkadot(vec2 texCoord)
{
	const vec3 bgColor = vec3(0.8, 0.8, 0.1);
	// This implementation is global, adding a set number of dots at random to the whole texture.
	prngSeed = globalPrngSeed; // Need to reseed here to play nicely with anti-aliasing
	const int nPolkaDots = 30;
	const float polkaDotRadius = 0.03;
	vec3 color = bgColor;
	
	#ifdef SOLUTION_TEXTURING
	#endif 
	return color;
}

vec3 textureVoronoi(vec2 texCoord)
{
	// This implementation is global, adding a set number of cells at random to the whole texture.
	prngSeed = globalPrngSeed; // Need to reseed here to play nicely with anti-aliasing
	const int nVoronoiCells = 15;
	
	#ifdef SOLUTION_TEXTURING
	#endif
	return vec3(0.0, 0.0, 1.0); 
}

vec3 getInterpVertexColor(Vertex interpVertex, int textureType)
{
	#ifdef SOLUTION_TEXTURING
	#else
	return interpVertex.color;
	#endif
	return vec3(1.0, 0.0, 1.0);
}

// Draws a polygon by projecting, clipping, ratserizing and interpolating it
void drawPolygon(
  vec2 point, 
  Polygon clipWindow, 
  Polygon oldPolygon, 
  inout vec3 color, 
  inout float depth)
{
    Polygon projectedPolygon;
    projectPolygon(projectedPolygon, oldPolygon);  
  
    Polygon clippedPolygon;
    sutherlandHodgmanClip(projectedPolygon, clipWindow, clippedPolygon);

    if (isPointInPolygon(point, clippedPolygon)) {
      
        Vertex interpolatedVertex = 
          interpolateVertex(point, projectedPolygon);
#ifdef SOLUTION_ZBUFFERING
#else
      color = getInterpVertexColor(interpolatedVertex, oldPolygon.textureType);
      depth = interpolatedVertex.position.z;      
#endif
   }
  
   if (isPointOnPolygonVertex(point, clippedPolygon)) {
        color = vec3(1);
   }
}

// Main function calls

void drawScene(vec2 pixelCoord, inout vec3 color) {
    color = vec3(0.3, 0.3, 0.3);
  
  	// Convert from GL pixel coordinates 0..N-1 to our screen coordinates -1..1
    vec2 point = 2.0 * pixelCoord / vec2(viewport) - vec2(1.0);

    Polygon clipWindow;
    clipWindow.vertices[0].position = vec4(-0.65,  0.95, 1.0, 1.0);
    clipWindow.vertices[1].position = vec4( 0.65,  0.75, 1.0, 1.0);
    clipWindow.vertices[2].position = vec4( 0.75, -0.65, 1.0, 1.0);
    clipWindow.vertices[3].position = vec4(-0.75, -0.85, 1.0, 1.0);
    clipWindow.vertexCount = 4;
	
	clipWindow.textureType = TEXTURE_NONE;
  
  	// Draw the area outside the clip region to be dark
    color = isPointInPolygon(point, clipWindow) ? vec3(0.5) : color;

    const int triangleCount = 3;
    Polygon triangles[triangleCount];
  
	triangles[0].vertexCount = 3;
    triangles[0].vertices[0].position = vec4(-3, -2, 0.0, 1.0);
    triangles[0].vertices[1].position = vec4(4, 0, 3.0, 1.0);
    triangles[0].vertices[2].position = vec4(-1, 2, 0.0, 1.0);
    triangles[0].vertices[0].color = vec3(1.0, 1.0, 0.2);
    triangles[0].vertices[1].color = vec3(0.8, 0.8, 0.8);
    triangles[0].vertices[2].color = vec3(0.5, 0.2, 0.5);
	triangles[0].vertices[0].texCoord = vec2(0.0, 0.0);
    triangles[0].vertices[1].texCoord = vec2(0.0, 1.0);
    triangles[0].vertices[2].texCoord = vec2(1.0, 0.0);
	triangles[0].textureType = TEXTURE_CHECKERBOARD;
  
	triangles[1].vertexCount = 3;
    triangles[1].vertices[0].position = vec4(3.0, 2.0, -2.0, 1.0);
  	triangles[1].vertices[2].position = vec4(0.0, -2.0, 3.0, 1.0);
    triangles[1].vertices[1].position = vec4(-1.0, 2.0, 4.0, 1.0);
    triangles[1].vertices[1].color = vec3(0.2, 1.0, 0.1);
    triangles[1].vertices[2].color = vec3(1.0, 1.0, 1.0);
    triangles[1].vertices[0].color = vec3(0.1, 0.2, 1.0);
	triangles[1].vertices[0].texCoord = vec2(0.0, 0.0);
    triangles[1].vertices[1].texCoord = vec2(0.0, 1.0);
    triangles[1].vertices[2].texCoord = vec2(1.0, 0.0);
	triangles[1].textureType = TEXTURE_POLKADOT;
	
	triangles[2].vertexCount = 3;	
	triangles[2].vertices[0].position = vec4(-1.0, -2.0, 0.0, 1.0);
  	triangles[2].vertices[1].position = vec4(-4.0, 2.0, 0.0, 1.0);
    triangles[2].vertices[2].position = vec4(-4.0, -2.0, 0.0, 1.0);
    triangles[2].vertices[1].color = vec3(0.2, 1.0, 0.1);
    triangles[2].vertices[2].color = vec3(1.0, 1.0, 1.0);
    triangles[2].vertices[0].color = vec3(0.1, 0.2, 1.0);
	triangles[2].vertices[0].texCoord = vec2(0.0, 0.0);
    triangles[2].vertices[1].texCoord = vec2(0.0, 1.0);
    triangles[2].vertices[2].texCoord = vec2(1.0, 0.0);
	triangles[2].textureType = TEXTURE_VORONOI;
	
    float depth = 10000.0;
    // Project and draw all the triangles
    for (int i = 0; i < triangleCount; i++) {
        drawPolygon(point, clipWindow, triangles[i], color, depth);
    }   
}

void main() {
	
	vec3 color = vec3(0);
	
#ifdef SOLUTION_AALIAS
#else
    drawScene(gl_FragCoord.xy, color);
#endif
	
	gl_FragColor.rgb = color;	
    gl_FragColor.a = 1.0;
}