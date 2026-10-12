def beta_schedule(epoch, epochs, beta_start=2.0, beta_end=20.0, warm_fraction=0.4):
    if epochs < 1 or not 1 <= epoch <= epochs or not 0 <= warm_fraction < 1:
        raise ValueError("Invalid epoch range or warm_fraction")
    progress = epoch / epochs
    if progress <= warm_fraction:
        return float(beta_start)
    return beta_start + (beta_end - beta_start) * (progress - warm_fraction) / (1 - warm_fraction)


def revisible_loss(blind, visible, noisy, alpha=1.0, beta=2.0):
    residual = blind - noisy
    correction = visible.detach() - noisy
    return alpha * residual.square().mean() + (residual + beta * correction).square().mean()
