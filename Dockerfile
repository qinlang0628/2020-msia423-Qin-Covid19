FROM ubuntu:18.04

RUN apt-get update -y && apt-get install -y curl python3 python3-pip python3-dev git gcc g++

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install pytest-docker-compose

COPY . /app

RUN chmod +x app/boot.sh

#RUN echo 'source .mysqlconfig' >> ~/.bashrc
#RUN source ~/.bashrc
#RUN chmod +x ./.mysqlconfig.sh

EXPOSE 5000

CMD ["./app/boot.sh"]
