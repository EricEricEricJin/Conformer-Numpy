import re

def get_mm_dict(fn):
    with open(fn, "r") as f:
        all_param = f.readlines()
        all_param = [i.strip(" \n") for i in all_param if i != "\n"]

    # all matmul
    all_mm = [int(re.findall("[0-9]+ ", i)[0]) for i in all_param if "MatMul" in i]

    # 145 = 1 + 16*9
    mm_iter = iter(all_mm)
    mm_dict = {"encoder.pre_encode": next(mm_iter)}

    for i in range(16):
        # feed forward
        mm_dict[f"encoder.layers.{i}"] = {}
        mm_dict[f"encoder.layers.{i}"]["feed_forward1.linear1.weight"] = next(mm_iter)
        mm_dict[f"encoder.layers.{i}"]["feed_forward1.linear2.weight"] = next(mm_iter)

        # self attention
        mm_dict[f"encoder.layers.{i}"]["self_attn.linear_q.weight"] = next(mm_iter)
        mm_dict[f"encoder.layers.{i}"]["self_attn.linear_k.weight"] = next(mm_iter)
        mm_dict[f"encoder.layers.{i}"]["self_attn.linear_v.weight"] = next(mm_iter)
        mm_dict[f"encoder.layers.{i}"]["self_attn.linear_pos.weight"] = next(mm_iter)
        mm_dict[f"encoder.layers.{i}"]["self_attn.linear_out.weight"] = next(mm_iter)

        # feed forward 2
        mm_dict[f"encoder.layers.{i}"]["feed_forward2.linear1.weight"] = next(mm_iter)
        mm_dict[f"encoder.layers.{i}"]["feed_forward2.linear2.weight"] = next(mm_iter)

    print(mm_dict["encoder.layers.0"])

    try: next(mm_iter)
    except StopIteration: print("End")

    return mm_dict


if __name__ == "__main__":
    mm_dict = get_mm_dict("param_shape.txt")
    print(mm_dict["encoder.layers.0"])