FROM python:3.11

WORKDIR .

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 50051
EXPOSE 8080

COPY . .

RUN chmod +x start.sh

CMD ["./run.sh"]