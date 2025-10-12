docker build -t model1_image:py3.13 ./model1

docker run --name model1_container -v "${PWD}/emissions_logs:/app/emissions_logs" model1_image:py3.13

docker build -t model2_image:py3.13 ./model2

docker run --name model2_container -v "${PWD}/emissions_logs:/app/emissions_logs" model2_image:py3.13