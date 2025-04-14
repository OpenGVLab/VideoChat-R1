DATASETS={
    'charades': {
        'video_path': 'your video root',
        'stride': 20,
        'max_stride_factor': 0.5,
        'splits': {
            'default': {
                'annotation_file': 'Annotations/Charades/charades_annotation/charades_test.json',
                'pad_sec': 0.0,
            }
        }
    },
    'activitynet': {
        'video_path': 'your video root',
        'stride': 40,
        'max_stride_factor': 1,
        'splits': {
            'default': {
                'annotation_file': 'Annotations/ActivityNet/activitynet_annotation/test.json',
                'pad_sec': 0.0,
            }
        }
    },
    'nextgqa': {
        'video_path': 'your video root',
        'stride': 40,
        'max_stride_factor': 1,
        'splits': {
            'default': {
                'annotation_file': 'Annotations/NextGQA/nextgqa_test.json',
                'pad_sec': 0.0,
            },
        }
    },
    'got': {
        'video_path': 'your video root',
        'stride': 40,
        'max_stride_factor': 1,
        'splits': {
            'default': {
                'annotation_file': 'Annotations/Got/got_val.json',
                'pad_sec': 0.0,
            },
        }
    }, 
    'cls_quality': {
        'video_path': 'your video root',
        'stride': 40,
        'max_stride_factor': 1,
        'splits': {
            'default': {
                'annotation_file': 'Annotations/VideoEval/Quality_Access/annotations/Quality_Access_test.json',
                'pad_sec': 0.0,
            },
        }
    }
}