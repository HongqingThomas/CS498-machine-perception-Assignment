Our dataset is a subset of 8 categories from the dataset described below:

This dataset provides part segmentation to a subset of ShapeNetCore models, containing ~16K models from 16 shape categories. The number of parts for each category varies from 2 to 6 and there are a total number of 50 parts.
The dataset is based on the following work:

@article{yi2016scalable,
  title={A scalable active framework for region annotation in 3d shape collections},
  author={Yi, Li and Kim, Vladimir G and Ceylan, Duygu and Shen, I and Yan, Mengyan and Su, Hao and Lu, ARCewu and Huang, Qixing and Sheffer, Alla and Guibas, Leonidas and others},
  journal={ACM Transactions on Graphics (TOG)},
  volume={35},
  number={6},
  pages={210},
  year={2016},
  publisher={ACM}
}

You could find the initial mesh files from the released version of ShapeNetCore.
An mapping from synsetoffset to category name could be found in "synsetoffset2category.txt"

The folder structure is as below:
	-synsetoffset
		-points
			-uniformly sampled points from ShapeNetCore models
		-point_labels
			-per-point segmentation labels
		-seg_img
			-a visualization of labeling
	-train_test_split
		-lists of training/test/validation shapes shuffled across all categories (from the official train/test split of ShapeNet)

