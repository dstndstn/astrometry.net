ALTER TABLE net_submission ADD COLUMN album_id integer;
ALTER TABLE net_album ADD COLUMN user_id integer REFERENCES auth_user(id);
ALTER TABLE net_album ADD COLUMN title varchar(64) NOT NULL;
ALTER TABLE net_userprofile ALTER COLUMN display_name TYPE varchar(64);


