import torch

def generate_orthogonal_anchor(d=256, existing_anchors=[], alpha_0=2.0):
    # 첫 번째 앵커 생성: Standard Normal Distribution
    if len(existing_anchors) == 0:
        epsilon = torch.randn(d)
        a_t = alpha_0 * (epsilon / torch.norm(epsilon, p=2))
        return a_t

    # 후속 앵커 생성: 유사도가 [-0.1, 0.1] 사이일 때까지 반복
    while True:
        epsilon = torch.randn(d)
        candidate_a_t = alpha_0 * (epsilon / torch.norm(epsilon, p=2))
        
        is_orthogonal = True
        for a_prev in existing_anchors:
            # Cosine Similarity 체크 [-0.1, 0.1]
            sim = torch.nn.functional.cosine_similarity(
                candidate_a_t.unsqueeze(0), 
                a_prev.to(candidate_a_t.device).unsqueeze(0) # 이 부분 추가!
            )
            if abs(sim) > 0.1:
                is_orthogonal = False
                break
        
        if is_orthogonal:
            return candidate_a_t