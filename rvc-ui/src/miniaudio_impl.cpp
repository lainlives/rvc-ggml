// miniaudio_impl.cpp — single translation unit that compiles the miniaudio and
// stb_vorbis implementations (decoding: WAV / FLAC / MP3 / Ogg Vorbis).
//
// miniaudio (https://github.com/mackron/miniaudio, MIT-0/public domain):
//   built-in decoders WAV/FLAC/MP3; Vorbis via the bundled stb_vorbis.
// Encoding through miniaudio is WAV-only; other output formats are encoded by
// ffmpeg in the UI layer.
#define STB_VORBIS_HEADER_ONLY
#include "stb_vorbis.c"
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#undef STB_VORBIS_HEADER_ONLY
#include "stb_vorbis.c"
