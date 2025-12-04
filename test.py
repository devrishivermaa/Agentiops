import fitz
import os

def is_caption(text):
    text_lower = text.lower().strip()
    return text_lower.startswith("figure") or text_lower.startswith("fig.")

def extract_figures(pdf_path, output_folder, dpi=300):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    results = {}

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page_index, page in enumerate(doc, start=1):
        print(f"\nProcessing page {page_index}")

        # Render full page as reference image
        pix = page.get_pixmap(matrix=matrix)
        full_img = pix.tobytes("png")
        full_image = fitz.Pixmap(full_img)

        blocks = page.get_text("dict")["blocks"]

        figure_blocks = []
        caption_blocks = []

        # Separate diagram blocks and caption blocks
        for b in blocks:
            if b["type"] == 0:   # text block
                text = ""
                for line in b["lines"]:
                    for span in line["spans"]:
                        text += span["text"] + " "

                if is_caption(text):
                    caption_blocks.append((b["bbox"], text.strip()))
            else:
                # non text block (vector drawings etc)
                figure_blocks.append(b["bbox"])

        # Merge figure blocks and caption into single region per figure
        page_results = []
        for cap_bbox, caption_text in caption_blocks:
            # find nearby diagram blocks
            merged_bbox = list(cap_bbox)

            for fig_bbox in figure_blocks:
                # expand region if overlapping or near
                if (
                    fig_bbox[0] < cap_bbox[2] and
                    fig_bbox[2] > cap_bbox[0] and
                    fig_bbox[1] < cap_bbox[3] + 200 and  # allow vertical margin
                    fig_bbox[3] > cap_bbox[1] - 200
                ):
                    merged_bbox[0] = min(merged_bbox[0], fig_bbox[0])
                    merged_bbox[1] = min(merged_bbox[1], fig_bbox[1])
                    merged_bbox[2] = max(merged_bbox[2], fig_bbox[2])
                    merged_bbox[3] = max(merged_bbox[3], fig_bbox[3])

            # Crop region
            rect = fitz.Rect(merged_bbox)
            pix_crop = page.get_pixmap(matrix=matrix, clip=rect)

            fig_filename = f"page_{page_index}_figure_{len(page_results)+1}.png"
            fig_path = os.path.join(output_folder, fig_filename)
            pix_crop.save(fig_path)

            # Save caption
            caption_filename = f"page_{page_index}_figure_{len(page_results)+1}_caption.txt"
            caption_path = os.path.join(output_folder, caption_filename)
            with open(caption_path, "w", encoding="utf8") as f:
                f.write(caption_text)

            print(f"Saved figure: {fig_filename}")
            print(f"Caption: {caption_text}")

            page_results.append({
                "image": fig_path,
                "caption": caption_text,
                "bbox": merged_bbox
            })

        results[f"page_{page_index}"] = page_results

    return results


if __name__ == "__main__":
    pdf_path = "C:\\Users\\devri\\OneDrive\\Desktop\\Agentiops\\data\\2510.02125v1.pdf"  # put your PDF file here
    output_folder = "figures_output"

    extract_figures(pdf_path, output_folder)
