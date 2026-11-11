// Reference driver around OpenCV 5.0.0's own stb_truetype.cpp (fetched by tools/stbtt_ref.sh): reads "codepoint size weight"
// lines on stdin, prints the glyph bitmap exactly as drawing_text.cpp asks for it (scale = ScaleForPixelHeightNoDesc).
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include "stb_truetype.hpp"
using namespace cv;
// usage: ref font.ttf  then rows "cp size weight" on stdin -> "xoff yoff w h adv\npix..."
int main(int argc, char** argv) {
  FILE* f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<unsigned char> buf(n); fread(buf.data(), 1, n, f); fclose(f);
  stbtt_fontinfo* ff = stbtt_CreateFont(buf.data(), (unsigned)n, stbtt_GetFontOffsetForIndex(buf.data(), 0));
  int cp, size, weight; unsigned char* gb = 0; int gbs = 0;
  while (scanf("%d %d %d", &cp, &size, &weight) == 3) {
    int params[] = {STBTT_FOURCC('w','g','h','t'), weight * 65536};
    stbtt_SetInstance(ff, params, 1, 0);
    float scale = stbtt_ScaleForPixelHeightNoDesc(ff, (float)size);
    int w=0,h=0,xoff=0,yoff=0,step=0; float advx=0;
    unsigned char* p = stbtt_GetGlyphBitmapSubpixelRealloc(ff, scale, scale, 0.f, 0.f, stbtt_FindGlyphIndex(ff, cp), &w,&h,&step,&xoff,&yoff,&advx,&gb,&gbs);
    printf("%d %d %d %d %d\n", xoff, yoff, w, h, (int)(lrintf(advx*64)) >> 6);
    for (int y=0;y<h;y++){ for(int x=0;x<w;x++) printf("%d ", p[y*step+x]); } printf("\n");
  }
}
