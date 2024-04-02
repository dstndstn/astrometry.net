from __future__ import print_function
import os
import sys
if __name__ == '__main__':
    me = __file__
    path = os.path.realpath(me)
    sys.path.append(os.path.dirname(os.path.dirname(path)))
os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'

from astrometry.net import settings
settings.DEBUG = True

import django
django.setup()
from astrometry.net.models import *
from astrometry.util.file import *

def clean_dfs():
    for df in DiskFile.objects.all().order_by('file_hash'):
        if os.path.exists(df.get_path()):
            continue
        print('Does not exist:', df)

        ocoll = df.collection
        for coll in ['cached', 'resized', 'uploaded', 'uploaded-gunzip', 'uploaded-untar']:
            df.collection = coll
            if os.path.exists(df.get_path()):
                print('--> found in', coll)
                df.save()
                continue

        df.delete()
        # print '  image_set:', df.image_set.all()
        # for im in df.image_set.all():
        #     print '    uis:', im.userimage_set.all()
        # print '  submissions:', df.submissions.all()
        # print '  cached:', df.cachedfile_set.all()


def unlink_resized_fits():
    uis = UserImage.objects.filter(image__disk_file__file_type='FITS image data')
    print(uis.count(), 'UserImages are FITS')
    for ui in uis:
        im = ui.image
        im.display_image = None
        im.thumbnail = None
        im.save()
    print('Updated', len(uis), 'UserImages')


def delete_orphaned_images():
    print('Checking for orphaned Images...')
    ndel = 0
    for im in Image.objects.all():
        used = (im.userimage_set.count() +
                im.image_thumbnail_set.count() +
                im.image_display_set.count())
        print('Image', im.id, 'used', used, 'times')
        if used > 0:
            continue
        im.delete()
        ndel += 1
    print('Deleted', ndel, 'Images')

def delete_orphaned_diskfiles():
    ndel = 0
    for df in DiskFile.objects.all():
        used = (df.image_set.count() + 
                df.submissions.count() +
                df.cachedfile_set.count())
        print('DiskFile', df.file_hash, 'used', used, 'times')
        if used > 0:
            continue
        os.remove(df.get_path())
        df.delete()
        ndel += 1
    print('Deleted', ndel, 'DiskFiles')



def clean_cache():
    cfs = CachedFile.objects.all()
    print(cfs.count(), 'CachedFiles')

    # These aren't produced any more...
    #cfs = cfs.filter(key__contains='galex')
    #print(cfs.count(), 'GALEX cached files')
    #cfs = cfs.filter(key__contains='sdss_size')
    #print(cfs.count(), 'SDSS cached files')

    cfs = cfs.filter(key__contains='jpg_image')
    print(cfs.count(), 'FITS->jpeg images')
    #cfs = cfs.filter(key__contains='fits_table_')
    #print(cfs.count(), 'FITS tables')

    def do_delete(delcfs, deldfs, delfiles):
        delcfs = list(delcfs)
        deldfs = list(deldfs)
        delfiles = list(delfiles)
        print('Total of', len(delcfs), 'CachedFiles to delete')
        print('Total of', len(delfiles), 'files to delete')
        print('Total of', len(deldfs), 'DiskFiles to delete')
        print('Deleting CachedFiles...')
        for cf in delcfs:
            cf.delete()
        print('Deleting DiskFiles...')
        for df in deldfs:
            df.delete()
        print('Deleting Files...')
        for fn in delfiles:
            if os.path.exists(fn):
                os.unlink(fn)
            else:
                print('File to be unlinked does not exist:', fn)
    delfiles = set()
    delcfs = set()
    deldfs = set()

    for i,cf in enumerate(cfs):
        if i % 1000 == 0:
            do_delete(delcfs, deldfs, delfiles)

            delfiles = set()
            delcfs = set()
            deldfs = set()

        print()
        print(cf.key)
        try:
            df = cf.disk_file
        except:
            print('DiskFile not found -- deleting CachedFile')
            delcfs.add(cf)
            continue
        path = df.get_path()
        print('->', path)
        print('Other CachedFiles sharing this DiskFile:')
        for ocf in df.cachedfile_set.all():
            if ocf.key != cf.key:
                print('  ', ocf.key)
                delcfs.add(ocf)
        delcfs.add(cf)
        deldfs.add(df)
        delfiles.add(path)

    do_delete(delcfs, deldfs, delfiles)

def clean_resized_images():

    #images = Image.objects.all()
    #print(images.count(), 'images')
    images = Image.objects.all()
    images = images.filter(disk_file__collection='resized')
    print(images.count(), 'in resized collection')
    
    #images = images[:1000]
    #print(images.count(), 'in resized collection')

    freed = 0
    
    #i = 0
    for i,img in enumerate(images):
    #for img in images:
        print()
        #i += 1
        print('Image', i+1, '-- freed', int(freed/1024/1024), 'MB so far')
        df = img.disk_file
        imgs = df.image_set.all()
        if len(imgs) != 1:
            print('Resized image', img, '-> disk file', df, '-> multiple images', imgs)
            continue
        print('Resized image', img, 'pixel size %i x %i' % (img.width, img.height))
        print(' -> disk file', df)
        print(' -> df path', df.get_path(), 'exists?', os.path.exists(df.get_path()))
        display_of = img.image_display_set.all()
        print(' -> display_set', display_of)
        for orig in display_of:
            print('    ->', orig, '-> display', orig.display_image)
        thumbnail_of = img.image_thumbnail_set.all()
        print(' -> thumbnail_set', thumbnail_of)
        print(' -> userimage_set', img.userimage_set.all())
        print(' -> display', img.display_image)
        print(' -> thumbnail', img.thumbnail)
        #print(' -> cachedfile_set', img.cachedfile_set)

        if (img.display_image is None
            and img.thumbnail is None
            and len(img.userimage_set.all()) == 0):

            if (len(thumbnail_of) == 0
                and len(display_of) == 1
                and display_of[0].display_image == img):
                print('Simple display-size image', img)

            elif (len(display_of) == 0
                and len(thumbnail_of) == 1
                and thumbnail_of[0].thumbnail == img):
                print('Simple thumbnail image', img)
                
            else:
                continue

            fn = df.get_path()
            if os.path.exists(fn):
                print('Deleting', fn)
                os.unlink(fn)
                freed += df.size
            print('Deleting', df)
            df.delete()
            print('Deleting', img)
            img.delete()
        
    # from django.db import connection
    # for q in connection.queries:
    #     print('SQL:', q)

    import numpy as np
    from django.db import connections
    times = []
    sqls = []
    for q in connections['default'].queries:
        times.append(float(q['time']))
        sqls.append(q['sql'])
    times = np.array(times)
    I = np.argsort(times)
    for i in I:
        if times[i] < 0.1:
            continue
        print('Time %.3f' % times[i], ':', sqls[i])

if __name__ == '__main__':

    #clean_cache()
    clean_resized_images()
    
    sys.exit(0)


    # Remove resized FITS image to retro-fix bug in an-fitstopnm
    unlink_resized_fits()

    # then remove orphaned Image objects
    delete_orphaned_images()
    # and orphaned DiskFiles
    delete_orphaned_diskfiles()

    # clean_dfs()
    # 
    # cfs = CachedFile.objects.all()
    # print 'Total of', cfs.count(), 'files cached'
    # nbytes = 0
    # for cf in cfs:
    #     df = cf.disk_file
    #     path = df.get_path()
    #     if not os.path.exists(path):
    #         print 'Path does not exist:', path
    #         print 'Other CachedFiles sharing this DiskFile:'
    #         df.cachedfile_set.all().delete()
    #         df.delete()
    #         #cf.delete()
    #         continue
    #     sz = file_size(path)
    #     print '  %-32s' % cf.key, '=>', path, '  (size: %i bytes)' % sz
    #     nbytes += sz
    # print 'Total of', nbytes, 'bytes'
        
def clean_cache():
    cfs = CachedFile.objects.all()
    print(cfs.count(), 'CachedFiles')
    cfs = cfs.filter(key__contains='galex')
    print(cfs.count(), 'GALEX cached files')

    delfiles = []
    delcfs = []
    deldfs = []

    for cf in cfs:
        print()
        print(cf.key)
        df = cf.disk_file
        path = df.get_path()
        print('->', path)
        print('Other CachedFiles sharing this DiskFile:')
        for ocf in df.cachedfile_set.all():
            print('  ', ocf.key)
            delcfs.append(ocf)
        delcfs.append(cf)
        deldfs.append(df)
        delfiles.append(path)

    delcfs = list(set(delcfs))
    deldfs = list(set(deldfs))
    delfiles = list(set(delfiles))
    print('Total of', len(delcfs), 'CachedFiles to delete')
    print('Total of', len(delfiles), 'files to delete')
    print('Total of', len(deldfs), 'DiskFiles to delete')
    for cf in delcfs:
        cf.delete()
    for df in deldfs:
        df.delete()
    for fn in delfiles:
        os.unlink(fn)

