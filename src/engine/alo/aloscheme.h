#ifndef ENGINE_ALO_ALOSCHEME_H
#define ENGINE_ALO_ALOSCHEME_H

namespace engine {
namespace alo {

enum ALOScheme {
    FAST,           ///< Legendre-Legendre (7,2,7)-27 - Fastest but less accurate
    ACCURATE,       ///< Legendre-TanhSinh (25,5,13)-1e-8 - Good balance of speed and accuracy
    HIGH_PRECISION  ///< TanhSinh-TanhSinh (10,30)-1e-10 - Highest accuracy but slower
};

} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_ALOSCHEME_H