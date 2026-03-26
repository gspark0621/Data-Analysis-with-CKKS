from desilofhe import Engine, Ciphertext
from util.keypack import KeyPack

def RotSum_LogN(engine: Engine, indicator_ct: Ciphertext, num_points: int, 
                keypack: KeyPack) -> Ciphertext:
    """
    O(log N) Prefix Sum (이웃 수 계산)
    """
    rot_Key = keypack.rotation_key
    if num_points <= 1:
        return indicator_ct
    
    result = indicator_ct
    shift = 1
    accumulated = 1
    
    while accumulated < num_points:
        step = min(shift, num_points - accumulated)
        
        # 오른쪽으로 shift 후 더하기
        shifted = engine.rotate(result, rot_Key, step)
        result = engine.add(result, shifted)
        
        accumulated += step
        shift *= 2
    
    return result
