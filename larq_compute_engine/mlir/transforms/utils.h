#ifndef LARQ_COMPUTE_ENGINE_MLIR_UTILS_H_
#define LARQ_COMPUTE_ENGINE_MLIR_UTILS_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir {

template <class TOp>
struct CleanupDeadOps : public OpRewritePattern<TOp> {
  using OpRewritePattern<TOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TOp op,
                                     PatternRewriter& rewriter) const override {
    if (op.use_empty()) {
      rewriter.eraseOp(op);
      return Pattern::matchSuccess();
    }
    return Pattern::matchFailure();
  }
};

}  // namespace mlir

#endif  // LARQ_COMPUTE_ENGINE_MLIR_UTILS_H_
