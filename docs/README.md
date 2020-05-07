# Blog!

You can edit the blog directly on the .md pages in [docs/\_posts](https://github.com/estberg/CSE481N/tree/master/docs/_posts) from the github web interface or locally just being sure to push to master. The site is generated from whatever content is on the master branch in this docs folder. It sometimes takes a few minutes to update after a push to master. If you want to do anything fancy and test locally, set up on your machine by following the instructions below.

## Set Up

### 
First, [install ruby, jekyll, and bundler](
https://jekyllrb.com/docs/installation/), if you do not have any of them. This should be easiest on a Mac.

From this directory, that is `docs`, run the following:

```
bundle config set path 'vendor/bundle' # gems just for this project
bundle install # installs gems
```

## Local Testing

To test locally, just run the following (there may be some warnings between `Generatting...` and `done`, don't worry).

```
$ bundle exec jekyll serve
> Configuration file: /Users/octocat/my-site/_config.yml
>            Source: /Users/octocat/my-site
>       Destination: /Users/octocat/my-site/_site
> Incremental build: disabled. Enable with --incremental
>      Generating...
>                    done in 0.309 seconds.
> Auto-regeneration: enabled for '/Users/octocat/my-site'
> Configuration file: /Users/octocat/my-site/_config.yml
>    Server address: http://127.0.0.1:4000/
>  Server running... press ctrl-c to stop.
```

To preview the site, in your web browser, navigate to http://localhost:4000.

## Documentation
Read about [posts](https://jekyllrb.com/docs/posts/) the main feature of Jekyll we use, or explore other features!