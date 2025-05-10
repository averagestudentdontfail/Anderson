// --- START OF FILE aloscheme.h ---

#ifndef ENGINE_ALO_ALOSCHEME_H
#define ENGINE_ALO_ALOSCHEME_H

namespace engine {
namespace alo {

enum ALOScheme {
    FAST,           ///< Legendre-Legendre (7,2,7)-27 - Fastest but less accurate (QuantLib nomenclature)
    ACCURATE,       ///< Legendre-TanhSinh (25,5,13)-1e-8 - Good balance (QuantLib nomenclature)
    HIGH_PRECISION  ///< TanhSinh-TanhSinh (10,30)-1e-10 - Highest accuracy (QuantLib nomenclature)
};

} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_ALOSCHEME_H
// --- END OF FILE aloscheme.h ---