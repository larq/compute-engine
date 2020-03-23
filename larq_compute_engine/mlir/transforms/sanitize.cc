#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "larq_compute_engine/mlir/transforms/utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace TFL {

namespace {

struct SanitizeLCE : public FunctionPass<SanitizeLCE> {
  void runOnFunction() override;
};

template <class TOp>
struct RenameLCE : public RewritePattern {
  RenameLCE(MLIRContext* context)
      : RewritePattern(std::string("tfl.UNSUPPORTED_custom_") +
                           std::string(TOp::getOperationName().drop_front(3)),
                       1, context) {}

  PatternMatchResult match(Operation* op) const override {
    // No need for further matches, we only care about the name
    return matchSuccess();
  }

  void rewrite(Operation* op, PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<TOp>(op, op->getResultTypes(),
                                     op->getOperands(), op->getAttrs());
  }
};

void SanitizeLCE::runOnFunction() {
  OwningRewritePatternList patterns;
  auto* ctx = &getContext();
  auto func = getFunction();

  patterns.insert<RenameLCE<TF::LceBsignOp>, RenameLCE<TF::LceBconv2dOp>,
                  mlir::CleanupDeadOps<TF::LceBsignOp>,
                  mlir::CleanupDeadOps<TF::LceBconv2dOp>>(ctx);

  applyPatternsGreedily(func, patterns);
}

}  // namespace

// Creates an instance of the SanitizeLCE pass.
std::unique_ptr<OpPassBase<FuncOp>> CreateSanitizeLCEPass() {
  return std::make_unique<SanitizeLCE>();
}

static PassRegistration<SanitizeLCE> pass(
    "tfl-sanitize-lce", "Sanitize LCE ops in TensorFlow Lite dialect");

}  // namespace TFL
}  // namespace mlir
