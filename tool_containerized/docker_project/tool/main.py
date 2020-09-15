import argparse
import json
from src.tracking.tracking import Tracker, delete_tmp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate tracks from video sequence.')
    parser.add_argument('--config', default='/data/configs/tracking_config_test.json', type=str,
                        help='Config file with the tracking parameters.')
    parser.add_argument('--save_vid', action='store_true', help='Save video with drawn tracks.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    tracker = Tracker(params=config)
    tracker.detect(detections_path=config['detections_csv'])
    tracker.track(detections_path=config['detections_csv'], tracks_path=config['tracks_csv'])
    tracker.get_who_measures(tracks_path=config['tracks_csv'], who_path=config['who_measures_csv'])
    tracker.save_vid(tracks_path=config['tracks_csv'], video_path=config['tracks_video'])
    delete_tmp()
