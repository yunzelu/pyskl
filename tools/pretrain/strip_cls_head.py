import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to NTU pretrained checkpoint")
    parser.add_argument("--dst", required=True, help="Path to save backbone-only checkpoint")
    args = parser.parse_args()

    ckpt = torch.load(args.src, map_location="cpu")

    if "state_dict" not in ckpt:
        raise KeyError("Checkpoint does not contain 'state_dict'")

    state_dict = ckpt["state_dict"]

    new_state_dict = {}
    removed_keys = []

    for k, v in state_dict.items():
        if k.startswith("cls_head."):
            removed_keys.append(k)
            continue
        new_state_dict[k] = v

    new_ckpt = {
        "state_dict": new_state_dict,
        "meta": ckpt.get("meta", {})
    }

    torch.save(new_ckpt, args.dst)

    print(f"Saved backbone-only checkpoint to: {args.dst}")
    print(f"Removed {len(removed_keys)} classification-head keys:")
    for k in removed_keys:
        print(f"  {k}")


if __name__ == "__main__":
    main()