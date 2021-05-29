# Run retrain
cd /home/admina/PyLeagueOracle/src
python3 retrain_best.py > ../retrain.log
cd ..

# Rebuild prediction image with new best model
docker build --tag pyleagueoracle .

# Stop and remove current prediction container
docker stop leaguePredict
docker rm leaguePredict

# Launch new prediction container
docker run -d --name leaguePredict --restart always --network="host" league-oracle-data-cull
