mkdir ./results
echo "*" >> ./results/.gitignore
python main.py train --network_name=mnist_mini_base --network_type=vnn --dataset_name=mnist --batch=20 --epochs=1 --activation=lrelu --activation_mode=mean --all_models_path=./results