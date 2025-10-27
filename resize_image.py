import argparse
from PIL import Image
import os

def make_square_image(img: Image.Image, mode: str = "crop", max_res: int = 1024) -> Image.Image:
    w, h = img.size
    if mode == "crop":
        min_side = min(w, h)
        left = (w - min_side) // 2
        top = (h - min_side) // 2
        img = img.crop((left, top, left + min_side, top + min_side))
    elif mode == "pad":
        max_side = max(w, h)
        new_img = Image.new(img.mode, (max_side, max_side), (0, 0, 0))
        left = (max_side - w) // 2
        top = (max_side - h) // 2
        new_img.paste(img, (left, top))
        img = new_img
    else:
        raise ValueError("mode는 'crop' 또는 'pad'만 가능합니다.")

    if img.size[0] > max_res or img.size[1] > max_res:
        img = img.resize((max_res, max_res), Image.LANCZOS)
    return img

def process_folder(input_folder, output_folder, mode, max_res):
    os.makedirs(output_folder, exist_ok=True)
    for fname in os.listdir(input_folder):
        fpath = os.path.join(input_folder, fname)
        if os.path.isfile(fpath) and fname.lower().endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(fpath)
            square_img = make_square_image(img, mode=mode, max_res=max_res)
            out_path = os.path.join(output_folder, fname)
            square_img.save(out_path)
            print(f"Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="이미지를 정방형으로 변환합니다.")
    parser.add_argument("input", help="입력 이미지 경로 또는 폴더")
    parser.add_argument("output", help="출력 이미지 경로 또는 폴더")
    parser.add_argument("--mode", choices=["crop", "pad"], default="pad", help="정방형 변환 방식 (crop 또는 pad)")
    parser.add_argument("--max-res", type=int, default=1024, help="최대 해상도 (기본값: 1024)")
    parser.add_argument("--folder", action="store_true", help="입력 경로가 폴더일 경우 폴더 전체 처리")
    args = parser.parse_args()

    if args.folder:
        process_folder(args.input, args.output, args.mode, args.max_res)
    else:
        # If output is a directory, use input filename
        if os.path.isdir(args.output):
            input_filename = os.path.basename(args.input)
            output_path = os.path.join(args.output, input_filename)
        else:
            output_path = args.output

        img = Image.open(args.input)
        square_img = make_square_image(img, mode=args.mode, max_res=args.max_res)
        square_img.save(output_path)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()