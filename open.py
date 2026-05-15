import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import scipy.io as sio


def _normalize_for_display(image):
	image = image.astype(float)
	image_min = image.min()
	image_max = image.max()
	if image_max > image_min:
		return (image - image_min) / (image_max - image_min)
	return image * 0


def _save_image(image, title, png_path):
	fig = plt.figure(figsize=(6, 6))
	plt.imshow(image, cmap='gray')
	plt.title(title)
	plt.axis('off')
	fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0)
	plt.close(fig)


def _save_merged_preview(original, gt, det, png_path):
	fig = plt.figure(figsize=(15, 5))
	for idx, (image, title) in enumerate([
		(original, 'Original image'),
		(gt, 'Ground truth'),
		(det, 'HSI anomaly'),
	], start=1):
		ax = fig.add_subplot(1, 3, idx)
		ax.imshow(image, cmap='gray')
		ax.set_title(title)
		ax.axis('off')
	fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0)
	plt.close(fig)


def main():
	parser = argparse.ArgumentParser(description='Visualize DFAN result .mat files.')
	parser.add_argument('--mat', type=str, default='Result/abu-airport-4.mat')
	parser.add_argument('--show_gt', action='store_true', help='Show gt map if available.')
	parser.add_argument('--out_dir', type=str, default='Result', help='Directory to save PNG files.')
	parser.add_argument('--show', action='store_true', help='Display figures after saving PNG files.')
	args = parser.parse_args()

	mat_path = Path(args.mat)
	if not mat_path.exists():
		raise FileNotFoundError(f'File not found: {mat_path}')

	mat_data = sio.loadmat(str(mat_path))
	original = mat_data.get('original')
	det = mat_data.get('det')
	auc_pd_pf = mat_data.get('auc_pd_pf')
	auc_pf_tau = mat_data.get('auc_pf_tau')
	if auc_pd_pf is not None and auc_pf_tau is not None:
		print('AUC: PD_PF_auc=%.5f / PF_tau_auc=%.5f' % (float(auc_pd_pf.squeeze()), float(auc_pf_tau.squeeze())))
	if det is None:
		raise KeyError("'det' not found in result file.")
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	if original is not None:
		original = _normalize_for_display(original.squeeze())
	else:
		original = None

	gt = mat_data.get('gt')
	if gt is not None:
		gt = _normalize_for_display(gt.squeeze())
	else:
		raise KeyError("'gt' not found in result file.")

	det = _normalize_for_display(det.squeeze())

	merged_png = out_dir / f'{mat_path.stem}_overview.png'
	if original is None:
		original = det
	_save_merged_preview(original, gt, det, merged_png)
	print(f'Saved: {merged_png}')

	if args.show:
		plt.show()
	else:
		plt.close('all')

if __name__ == '__main__':
	main()