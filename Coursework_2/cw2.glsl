#define SOLUTION_RASTERIZATION
#define SOLUTION_CLIPPING
#define SOLUTION_INTERPOLATION
#define SOLUTION_ZBUFFERING
#define SOLUTION_AALIAS
#define SOLUTION_TEXTURING

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
    // This function assumes that the segments are not parallel or collinear.

    vec2 p, w, p1w1, p1w2, w1p1, w1p2;
    p.xy = poli2.position.xy - poli1.position.xy;
    w.xy = wind2.position.xy - wind1.position.xy;
    p1w1.xy = wind1.position.xy - poli1.position.xy;
    p1w2.xy = wind2.position.xy - poli1.position.xy;
    w1p1.xy = poli1.position.xy - wind1.position.xy;
    w1p2.xy = poli2.position.xy - wind1.position.xy;

    float testw1 = p.x * p1w1.y - p1w1.x * p.y;
    float testw2 = p.x * p1w2.y - p1w2.x * p.y;
    float testp1 = w.x * w1p1.y - w1p1.x * w.y;
    float testp2 = w.x * w1p2.y - w1p2.x * w.y;

    if (testw1 * testw2 <= 0. && testp1 * testp2 <= 0.) {
        if (testp1 <= 0.) {
            return LEAVING;
        } else {
            return ENTERING;
        }
    } else if (testp2 < 0.) {
        return INSIDE;
    } else {
        return OUTSIDE;
    }
    
#else
    return INSIDE;
#endif
}
  
// This function assumes that the segments are not parallel or collinear.
Vertex intersect2D(Vertex a, Vertex b, Vertex c, Vertex d) {
#ifdef SOLUTION_CLIPPING
    // TODO
    
    vec2 A, B, U;
    A.xy = b.position.xy - a.position.xy;
    B.xy = d.position.xy - c.position.xy;
    U.xy = b.position.xy - d.position.xy;
    float UxB = U.x * B.y - B.x * U.y, AxB = A.x * B.y - B.x * A.y;
    float T = UxB / AxB;

    Vertex E;
    E.position = b.position;
    E.position.xy -= T * A.xy;
    T = length(E.position.xy - A.xy) / length(B.xy - A.xy);
    E.position.z = 1. / (1. / a.position.z + T * (1. / b.position.z - 1. / a.position.z));

    E.color = T * b.color + (1. - T) * a.color;
    E.texCoord = T * a.texCoord + (1. - T) * b.texCoord;

    return E;

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
            
            Vertex poli1, poli2, wind1, wind2;
            poli1 = getWrappedPolygonVertex(oldClipped, j);
            poli2 = getWrappedPolygonVertex(oldClipped, j + 1);
            wind1 = getWrappedPolygonVertex(clipWindow, i);
            wind2 = getWrappedPolygonVertex(clipWindow, i + 1);

            // Check if the segments are collinear.
            vec2 p1w1, p1w2, p2w1, p2w2;
            p1w1.xy = wind1.position.xy - poli1.position.xy;
            p1w2.xy = wind2.position.xy - poli1.position.xy;
            p2w1.xy = wind1.position.xy - poli2.position.xy;
            p2w2.xy = wind2.position.xy - poli2.position.xy;

            vec2 p1, p2, w1, w2;
            p1.xy = poli1.position.xy;
            p2.xy = poli2.position.xy;
            w1.xy = wind1.position.xy;
            w2.xy = wind2.position.xy;
            
            float maxx = w1.x > w2.x ? w1.x : w2.x;
            float minx = w1.x > w2.x ? w2.x : w1.x;
            float maxy = w1.y > w2.y ? w1.y : w2.y;
            float miny = w1.y > w2.y ? w2.y : w1.y;

            float testp1 = abs(p1w1.x * p1w2.y - p1w2.x * p1w1.y);
            float testp2 = abs(p2w1.x * p2w2.y - p2w2.x * p2w1.y);
            if (testp1 < 0.001 && testp2 < 0.001) {
                //if (minx <= p1.x && p1.x <= maxx && miny <= p1.y && p1.y <= maxy) {
                //    appendVertexToPolygon(clipped, poli1);    
                //}
                if (minx <= p2.x && p2.x <= maxx && miny <= p2.y && p2.y <= maxy) {
                    appendVertexToPolygon(clipped, poli2);    
                }
                continue;
            }

            int clipType = getCrossType(poli1, poli2, wind1, wind2);
            if (clipType == ENTERING) {
                appendVertexToPolygon(clipped, intersect2D(poli1, poli2, wind1, wind2));
                appendVertexToPolygon(clipped, poli2);
            } else if (clipType == LEAVING) {
                appendVertexToPolygon(clipped, intersect2D(poli1, poli2, wind1, wind2)); 
            } else if (clipType == INSIDE) {
                appendVertexToPolygon(clipped, poli2);
            }
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
    ab.xy = b.position.xy - a.position.xy;
    ap.xy = point.xy - a.position.xy;
    if (ab.x * ap.y - ap.x * ab.y <= 0.) {
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
            Vertex A = getWrappedPolygonVertex(polygon, 0);
            positionSum = A.position;
            positionSum.xy = point.xy;
#endif

#ifdef SOLUTION_ZBUFFERING
            // TODO
            // In my realization, no code is needed here.
#endif

#ifdef SOLUTION_INTERPOLATION
            // TODO
            Vertex M, N;
            M = getWrappedPolygonVertex(polygon, i + polygon.vertexCount - 1);
            A = getWrappedPolygonVertex(polygon, i);
            N = getWrappedPolygonVertex(polygon, i + 1);
            vec2 Axy, Mxy, Nxy, PA, PM, PN;
            Axy.xy = A.position.xy;
            Mxy.xy = M.position.xy;
            Nxy.xy = N.position.xy;
            PA = Axy - point;
            PM = Mxy - point;
            PN = Nxy - point;
            float alpha, beta, cosalpha, cosbeta;
            cosalpha = dot(PM, PA) / length(PM) / length(PA);
            cosbeta  = dot(PN, PA) / length(PN) / length(PA);
            alpha = acos(cosalpha);
            beta  = acos(cosbeta);
            float wi = (tan(alpha / 2.) + tan(beta / 2.)) / length(PA);
            weight_sum += wi;
            colorSum += A.color * wi;
#endif

#ifdef SOLUTION_TEXTURING
            texCoordSum += A.texCoord * wi;
            weight_corr_sum += wi;
#endif
        }
    }
    Vertex result = polygon.vertices[0];
  
#ifdef SOLUTION_INTERPOLATION
    // TODO
    result.color = colorSum / weight_sum;
    for (int i = 0; i < MAX_VERTEX_COUNT; i++) {
        if (i == polygon.vertexCount) {
            break;
        }
        Vertex B = getWrappedPolygonVertex(polygon, i);
        Vertex C = getWrappedPolygonVertex(polygon, i + 1);
        vec2 BP = point.xy - B.position.xy;
        vec2 CP = point.xy - C.position.xy;
        // If p is on the edge
        if (abs(BP.x * CP.y - CP.x * BP.y) < 0.01) {
            float T = length(point.xy - B.position.xy) / length(C.position.xy - B.position.xy);
            result.color = T * C.color + (1. - T) * B.color;
            break;
        }
    }
#endif
#ifdef SOLUTION_ZBUFFERING
    // TODO
    Vertex A = getWrappedPolygonVertex(polygon, 0);
    Vertex B = getWrappedPolygonVertex(polygon, 1);
    Vertex C = getWrappedPolygonVertex(polygon, polygon.vertexCount - 1);
    vec2 AP = point.xy - A.position.xy;
    vec2 BC = C.position.xy - B.position.xy;
    vec2 CP = point.xy - C.position.xy;
    float UxB = CP.x * BC.y - BC.x * CP.y, AxB = AP.x * BC.y - BC.x * AP.y;
    float T = UxB / AxB;
    vec2 E = point - T * AP;
    float s = length(E - B.position.xy) / length(C.position.xy - B.position.xy);
    float Ez = 1. / (1. / B.position.z + s * (1. / C.position.z - 1. / B.position.z));
    s = length(point.xy - A.position.xy) / length(E - A.position.xy);
    positionSum.z = 1. / (1. / A.position.z + s * (1. / Ez - 1. / A.position.z));
    result.position = positionSum;
#endif

#ifdef SOLUTION_TEXTURING
    result.texCoord = texCoordSum / weight_corr_sum;
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
    int x = int(texCoord.x * 8.), y = int(texCoord.y * 8.);
    if ((x + y) / 2 * 2 != x + y) {
        return vec3(0.6, 0.6, 0.6);
    } else {
        return vec3(1.0, 1.0, 1.0);
    }
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
    for (int i = 0; i < nPolkaDots; i++) {
        vec2 center = vec2(prngUniform01(), prngUniform01());
        vec3 dotColor = randomColor();
        if (length(center - texCoord) <= polkaDotRadius) {
            return dotColor;
        }
    }
	#endif 
	return color;
}

vec3 textureVoronoi(vec2 texCoord)
{
	// This implementation is global, adding a set number of cells at random to the whole texture.
	prngSeed = globalPrngSeed; // Need to reseed here to play nicely with anti-aliasing
	const int nVoronoiCells = 15;
    vec2 VoronoiPoints[nVoronoiCells];
    vec3 colors[nVoronoiCells];
    for (int i = 0; i < nVoronoiCells; i++) {
        VoronoiPoints[i] = vec2(prngUniform01(), prngUniform01());
        colors[i] = randomColor();
    }
    for (int k = 0; k < nVoronoiCells; k++) {
        float dk = length(VoronoiPoints[k] - texCoord);
        bool inK = true;
        for (int j = 0; j < nVoronoiCells; j++) {
            if (j != k) {
                float dj = length(VoronoiPoints[j] - texCoord);
                if (dk > dj) {
                    inK = false;
                    break;
                }
            }
        }
        if (inK) {
            return colors[k];
        }
    }

	#ifdef SOLUTION_TEXTURING
	#endif
	return vec3(0.0, 0.0, 1.0); 
}

vec3 getInterpVertexColor(Vertex interpVertex, int textureType)
{
	#ifdef SOLUTION_TEXTURING
        if (textureType == TEXTURE_CHECKERBOARD) {
            return textureCheckerboard(interpVertex.texCoord);
        } else if (textureType == TEXTURE_POLKADOT) {
            return texturePolkadot(interpVertex.texCoord);
        } else if (textureType == TEXTURE_VORONOI) {
            return textureVoronoi(interpVertex.texCoord);
        } else {
           return interpVertex.color; 
        }
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
        float now_depth = interpolatedVertex.position.z;
        if (now_depth < depth) {
            color = getInterpVertexColor(interpolatedVertex, oldPolygon.textureType);
            depth = now_depth;    
        }
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
    
    // SSAA
    /* The bigger zoom is, the better the effect of anto-aliasing is, 
     * but the speed of rendering will be slower.
     */
    const int zoom = 2; 
    vec4 old_coord = gl_FragCoord;
    vec3 colorSum = vec3(0.);
    for (int i = 0; i < zoom; i++) {
        for (int j = 0; j < zoom; j++) {
            vec4 now_coord = old_coord;
            now_coord.x = old_coord.x * float(zoom) + float(i);
            now_coord.y = old_coord.y * float(zoom) + float(j);
            drawScene(now_coord.xy / float(zoom), color);
            colorSum += color;
        }
    }
    color = colorSum / (float(zoom) * float(zoom));
#else
    drawScene(gl_FragCoord.xy, color);
#endif
	
	gl_FragColor.rgb = color;	
    gl_FragColor.a = 1.0;
}