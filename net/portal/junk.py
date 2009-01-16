class JunkInTheTrunk(object):
    # Choose a new unique fileid that does not conflict with an existing
    # file.  May have the side-effect of calling save().
    def choose_new_fileid(self):
        if not self.fileid:
            # find the largest existing fileid and add 1.
            try:
                v = UserFile.objects.filter(fileid__isnull=False).order_by('-fileid').values('fileid')[0]
                v = v['fileid']
                self.fileid = v + 1
            except IndexError:
                self.fileid = 1
        while True:
            log('Trying fileid %i...' % self.fileid)
            # save() to indicate that we are going to try to use this fileid.
            self.save()
            # check if any other UserFile has this fileid:
            n = UserFile.objects.filter(fileid=self.fileid).count()
            if n > 1:
                # some other UserFile already has this fileid.
                log('UserFile database: %i entries have fileid %i' % (n, self.fileid))
                self.fileid += 1
                continue
            # check if the file already exists.
            fn = UserFile.get_filename_for_fileid(self.fileid)
            if os.path.exists(fn):
                # the file already exists.
                log('UserFile: file %s already exists for fileid %i' % (fn, self.fileid))
                self.fileid += 1
                continue
            # touch the file to claim it.
            try:
                f = open(fn, 'wb')
                f.close()
            except IOError:
                # error touching the file.
                log('UserFile: error touching file %s for fileid %i' % (fn, self.fileid))
                self.fileid += 1
                continue
            break

    def redistributable(self, prefs=None):
        if self.allowredist:
            return True
        if self.forbidredist:
            return False
        if not prefs:
            prefs = self.user.get_profile()
        return prefs.autoredistributable

    def compute_filehash(self, fn):
        if self.filehash:
            return
        h = sha.new()
        f = open(fn, 'rb')
        while True:
            d = f.read(4096)
            if len(d) == 0:
                break
            h.update(d)
        self.filehash = h.hexdigest()

    def filename(self):
        if not self.fileid:
            self.choose_new_fileid()
        return UserFile.get_filename_for_fileid(self.fileid)

    def get_filename_for_fileid(fileid):
        return os.path.join(config.fielddir, str(fileid))
    get_filename_for_fileid = staticmethod(get_filename_for_fileid)

