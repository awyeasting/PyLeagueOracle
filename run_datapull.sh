docker build --tag league-oracle-data-pull -f Dockerfile.pull .
docker run -d --name leaguePull --restart always --network="host" league-oracle-data-pull
