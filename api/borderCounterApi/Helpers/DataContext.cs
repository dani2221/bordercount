using System;
using borderCounterApi.Models;
using Microsoft.EntityFrameworkCore;

namespace borderCounterApi.Helpers
{
	public class DataContext : DbContext
	{
        protected readonly IConfiguration Configuration;

        public DataContext(IConfiguration configuration)
        {
            Configuration = configuration;
        }

        protected override void OnConfiguring(DbContextOptionsBuilder options)
        {
            options.UseNpgsql(Configuration.GetConnectionString("WebApiDatabase"));
        }

        public DbSet<BorderInformationInterval> BorderInformationIntervals { get; set; }
    }
}

