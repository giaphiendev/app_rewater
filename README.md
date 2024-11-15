# Labeling use [roboflow](https://app.roboflow.com/rewaterplastic?skipSetup=true)

# Model use [YOLO](https://docs.ultralytics.com/modes/train/#introduction)



# Create migration file
$ docker exec -it app_rewater bash
$ alembic revision --autogenerate -m "init"

# Migrate newest migration file
$ docker exec -it app_rewater bash
$ alembic upgrade head

# Rollback to previous migration
$ docker exec -it app_rewater bash
$ alembic downgrade -1
