
def erode_detection(img_bin: np.ndarray, templates: list) -> bool:
    """
    Use binary erosion: if any rotated SE fits fully, detection is True.
    """
    for tmpl in templates:
        # Erode with the structuring element
        eroded = cv2.erode(img_bin, tmpl)
        if cv2.countNonZero(eroded) > 0:
            return True
    return False


def detect_symbols(img_bin: np.ndarray, se_templates: dict, thresh: float = 0.7) -> dict:
    """
    Detect swastika and SS symbols in a binarized image.
    Returns a dict with detection flags and matching scores.
    """
    results = {}
    for name, tmpls in se_templates.items():
        mt_flag, score = match_template(img_bin, tmpls, thresh)
        er_flag = erode_detection(img_bin, tmpls)
        detected = mt_flag or er_flag
        results[name] = {
            'detected': detected,
            'template_score': score,
            'eroded_fit': er_flag
        }
    return results


def evaluate_predictions(df: pd.DataFrame) -> None:
    """
    Compute and print performance metrics comparing labels vs predictions.
    Expects columns: 'label' and 'predicted' with values 0 or 1.
    """
    y_true = df['label'].astype(int)
    y_pred = df['predicted'].astype(int)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Symbol', 'Symbol']))


# def main(image_dir: str, se_dir: str, csv_path: str, thresh: float = 0.7) -> None:
#     # Load SE templates
#     se_templates = load_structural_elements(se_dir)
#
#     # Read labeled dataset
#     df_labels = pd.read_csv(csv_path)
#     results = []
#
#     # Process each image in CSV
#     for _, row in df_labels.iterrows():
#         fname = row['filename']
#         label = row['label']  # 1 if symbol present, 0 otherwise
#         img_path = os.path.join(image_dir, fname)
#         if not os.path.isfile(img_path):
#             print(f"Warning: {img_path} not found, skipping.")
#             continue
#
#         # Preprocess and detect
#         img_bin = preprocess_meme(img_path)
#         dets = detect_symbols(img_bin, se_templates, thresh)
#
#         # Decide predicted label: any symbol detected
#         pred = 1 if any(d['detected'] for d in dets.values()) else 0
#
#         results.append({
#             'filename': fname,
#             'label': label,
#             'predicted': pred,
#             'swastika_detected': int(dets['swastika']['detected']),
#             'ss_detected': int(dets['ss']['detected']),
#             'sw_score': dets['swastika']['template_score'],
#             'ss_score': dets['ss']['template_score'],
#             'sw_eroded_fit': int(dets['swastika']['eroded_fit']),
#             'ss_eroded_fit': int(dets['ss']['eroded_fit'])
#         })

    # # Save results and evaluate
    # df_res = pd.DataFrame(results)
    # out_csv = 'detection_results.csv'
    # df_res.to_csv(out_csv, index=False)
    # print(f"Results written to {out_csv}")
    # evaluate_predictions(df_res)

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description='Detect hateful symbols (swastika, SS) in images using morphological operations and template matching')
#     parser.add_argument('--images', required=True, help='Folder with input images')
#     parser.add_argument('--templates', required=True, help='Folder with SE template images')
#     parser.add_argument('--csv', required=True, help='CSV file with columns [filename,label]')
#     parser.add_argument('--threshold', type=float, default=0.7, help='Template matching threshold')
#     args = parser.parse_args()
#     main(args.images, args.templates, args.csv, args.threshold)
